
import sys

from videocore6.assembler import assemble, qpu


@qpu
def qpu_ctbmv(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: True):

    g = globals()
    for i, v in enumerate(['length', 'x', 'x_inc', 'y_src', 'y_dst', 'y_inc',
                           'qpu_num']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_x_inc))
    nop(sig=ldunifrf(reg_y_src))
    nop(sig=ldunifrf(reg_y_inc))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 8 * (thread_num + 16 * qpu_num) * inc
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 3)
    umul24(r1, r0, reg_x_inc)
    add(reg_x, reg_x, r1).umul24(r1, r0, reg_y_inc)
    add(reg_y_src, reg_y_src, r1).add(reg_y_dst, reg_y_src, r1)

    # inc *= 8 * 16 * num_qpus
    mov(r0, 1)
    shl(r0, r0, 3 + 4 + num_qpus_shift)
    umul24(reg_x_inc, reg_x_inc, r0)
    umul24(reg_y_inc, reg_y_inc, r0)

    # length /= 16 * 2 * num_qpus * unroll
    shr(reg_length, reg_length, 4 + 1 + num_qpus_shift + unroll_shift)

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(2):
            mov(tmua, reg_x)
            mov(tmua, reg_y_src).mov(r5, 4)
            add(tmua, reg_x, r5).add(reg_x, reg_x, reg_x_inc)
            add(tmua, reg_y_src, r5).add(reg_y_src, reg_y_src, reg_y_inc)

        for j in range(unroll - 1):
            for i in range(2):
                # Load xr, yr, xi, yi in this order.
                # (xr + xi i) (yr + yi i) = (xr yr - xi yi) + (xr yi + yr xi) i
                nop(sig=ldtmu(r0))
                nop(sig=ldtmu(r1))
                fmul(r4, r0, r1, sig=ldtmu(r2))
                fmul(r1, r1, r2, sig=ldtmu(r3))
                mov(tmua, reg_x).fmul(r2, r2, r3)
                fsub(tmud, r4, r2).fmul(r0, r0, r3)
                mov(tmua, reg_y_dst)
                fadd(tmud, r0, r1).mov(r5, 4)
                add(tmua, reg_y_dst, r5).add(reg_y_dst, reg_y_dst, reg_y_inc)
                mov(tmua, reg_y_src)
                add(tmua, reg_x, r5).add(reg_x, reg_x, reg_x_inc)
                add(tmua, reg_y_src, r5).add(reg_y_src, reg_y_src, reg_y_inc)

        nop(sig=ldtmu(r0))
        nop(sig=ldtmu(r1))
        fmul(r4, r0, r1, sig=ldtmu(r2))
        fmul(r1, r1, r2, sig=ldtmu(r3))
        fmul(r2, r2, r3)
        fsub(tmud, r4, r2).fmul(r0, r0, r3)
        mov(tmua, reg_y_dst)
        fadd(tmud, r0, r1).mov(r5, 4)
        add(tmua, reg_y_dst, r5).add(reg_y_dst, reg_y_dst, reg_y_inc)

        nop(sig=ldtmu(r0))
        nop(sig=ldtmu(r1))
        fmul(r4, r0, r1, sig=ldtmu(r2))
        fmul(r1, r1, r2, sig=ldtmu(r3))
        sub(reg_length, reg_length, 1, cond='pushz').fmul(r2, r2, r3)
        fsub(tmud, r4, r2).fmul(r0, r0, r3)

        l.b(cond='na0')
        mov(tmua, reg_y_dst)
        fadd(tmud, r0, r1).mov(r5, 4)
        add(tmua, reg_y_dst, r5).add(reg_y_dst, reg_y_dst, reg_y_inc)

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def main():

    num_qpus, unroll_shift, code_offset = map(int, sys.argv[1:])

    for insn in assemble(qpu_ctbmv, num_qpus=num_qpus,
                         unroll_shift=unroll_shift, code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
