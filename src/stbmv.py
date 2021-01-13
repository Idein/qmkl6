
import sys

from videocore6.assembler import assemble, qpu


@qpu
def qpu_stbmv(asm, *, num_qpus, unroll_shift, code_offset,
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

    # addr += 4 * (thread_num + 16 * qpu_num) * inc
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    umul24(r1, r0, reg_x_inc)
    add(reg_x, reg_x, r1).umul24(r1, r0, reg_y_inc)
    add(reg_y_src, reg_y_src, r1).add(reg_y_dst, reg_y_src, r1)

    # inc *= 4 * 16 * num_qpus
    mov(r0, 1)
    shl(r0, r0, 6 + num_qpus_shift)
    umul24(reg_x_inc, reg_x_inc, r0)
    umul24(reg_y_inc, reg_y_inc, r0)

    # length /= 16 * 4 * num_qpus * unroll
    shr(reg_length, reg_length, 4 + 2 + num_qpus_shift + unroll_shift)

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(4):
            mov(tmua, reg_x).add(reg_x, reg_x, reg_x_inc)
            mov(tmua, reg_y_src).add(reg_y_src, reg_y_src, reg_y_inc)

        for j in range(unroll - 1):
            for i in range(4):
                nop(sig=ldtmu(r0))
                nop(sig=ldtmu(r1))
                fmul(tmud, r0, r1)
                mov(tmua, reg_y_dst).add(reg_y_dst, reg_y_dst, reg_y_inc)
                mov(tmua, reg_x).add(reg_x, reg_x, reg_x_inc)
                mov(tmua, reg_y_src).add(reg_y_src, reg_y_src, reg_y_inc)

        for i in range(2):
            nop(sig=ldtmu(r0))
            nop(sig=ldtmu(r1))
            fmul(tmud, r0, r1)
            mov(tmua, reg_y_dst).add(reg_y_dst, reg_y_dst, reg_y_inc)

        nop(sig=ldtmu(r0))
        nop(sig=ldtmu(r1))
        sub(reg_length, reg_length, 1, cond='pushz').fmul(tmud, r0, r1)
        mov(tmua, reg_y_dst).add(reg_y_dst, reg_y_dst, reg_y_inc)

        nop(sig=ldtmu(r0))

        l.b(cond='na0')
        nop(sig=ldtmu(r1))                                         # delay slot
        fmul(tmud, r0, r1)                                         # delay slot
        mov(tmua, reg_y_dst).add(reg_y_dst, reg_y_dst, reg_y_inc)  # delay slot

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

    for insn in assemble(qpu_stbmv, num_qpus=num_qpus,
                         unroll_shift=unroll_shift, code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
