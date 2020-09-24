
import sys

from videocore6.assembler import assemble, qpu


@qpu
def qpu_sscal(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: pos % 512 == 259):

    acc_src = r0
    acc_dst = r1
    acc_stride = r2
    acc_length = r3
    acc_a = r4

    # r5 = qpu_num
    if num_qpus == 1:
        num_qpus_shift = 0
        mov(r5, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r5)
        shr(r5, r5, 2)
        band(r5, r5, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * (thread_num + 16 * qpu_num) * inc
    eidx(r4)
    shl(r5, r5, 4)
    add(r4, r4, r5, sig=ldunifrf(acc_stride))
    shl(r4, r4, 2)
    umul24(acc_src, r4, acc_stride, sig=ldunif)
    add(acc_src, r5, acc_src).add(acc_dst, r5, acc_src)

    # stride = 4 * 16 * num_qpus * inc
    mov(r4, 1)
    shl(r4, r4, 2 + 4 + num_qpus_shift)
    umul24(acc_stride, r4, acc_stride)

    # length /= 64 * 16 * num_qpus
    nop(sig=ldunifrf(acc_length))
    shr(acc_length, acc_length, 6 + 4 + num_qpus_shift + unroll_shift)

    nop(sig=ldunifrf(acc_a))

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        for i in range(8):
            mov(tmua, acc_src).add(acc_src, acc_src, acc_stride)

        for i in range(64 - 8):
            mov(tmua, acc_src, sig=ldtmu(rf[i])) \
                .add(acc_src, acc_src, acc_stride)

        for i in range(8):
            nop(sig=ldtmu(rf[64 - 8 + i]))

        for i in range(64 - 2):
            fmul(tmud, acc_a, rf[i])
            mov(tmua, acc_dst).add(acc_dst, acc_dst, acc_stride)

        sub(acc_length, acc_length, 1, cond='pushz').fmul(tmud, acc_a, rf[62])
        l.b(cond='na0')
        mov(tmua, acc_dst).add(acc_dst, acc_dst, acc_stride)
        fmul(tmud, acc_a, rf[63])
        mov(tmua, acc_dst).add(acc_dst, acc_dst, acc_stride)

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

    for insn in assemble(qpu_sscal, num_qpus=num_qpus,
                         unroll_shift=unroll_shift, code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
