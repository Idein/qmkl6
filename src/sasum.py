
import sys

from videocore6.assembler import assemble, qpu


@qpu
def qpu_sasum(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: pos % 512 == 170):

    g = globals()
    for i, v in enumerate(['length', 'x', 'x_inc', 'dst', 'qpu_num']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_x_inc))
    nop(sig=ldunifrf(reg_dst))

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
    # dst += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_dst, reg_dst, r0).umul24(r1, r0, reg_x_inc)
    add(reg_x, reg_x, r1)

    # inc *= 4 * 16 * num_qpus
    mov(r0, 1)
    shl(r0, r0, 6 + num_qpus_shift)
    umul24(reg_x_inc, reg_x_inc, r0)

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, 4 + 3 + num_qpus_shift + unroll_shift)

    nop(sig=thrsw)
    nop()
    bxor(r3, 1, 1).mov(r2, 1)

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(7):
            mov(tmua, reg_x).add(reg_x, reg_x, reg_x_inc)

        sub(reg_length, reg_length, r2, cond='pushz').mov(tmua, reg_x)
        add(reg_x, reg_x, reg_x_inc, sig=ldtmu(r0))

        for j in range(unroll - 1):
            for i in range(8):
                mov(tmua, reg_x).add(reg_x, reg_x, reg_x_inc)
                fadd(r3, r3, r0.unpack('abs'), sig=ldtmu(r0))

        for i in range(5):
            fadd(r3, r3, r0.unpack('abs'), sig=ldtmu(r0))

        l.b(cond='na0')
        fadd(r3, r3, r0.unpack('abs'), sig=ldtmu(r0))  # delay slot
        fadd(r3, r3, r0.unpack('abs'), sig=ldtmu(r0))  # delay slot
        fadd(r3, r3, r0.unpack('abs'))                 # delay slot

    mov(tmud, r3)
    mov(tmua, reg_dst)

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

    for insn in assemble(qpu_sasum, num_qpus=num_qpus,
                         unroll_shift=unroll_shift, code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
