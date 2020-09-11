
import sys

from videocore6 import pack_unpack
from videocore6.assembler import assemble, qpu


@qpu
def qpu_sgemv_n(asm, *, num_qpus, code_offset, align_cond=lambda pos: True):

    IDX0_INCX, IDX0_X_ADDR, IDX0_INCY, IDX0_Y_ADDR, IDX0_LDA, IDX0_A_ADDR, \
        IDX0_M, IDX0_N, IDX0_ALPHA, IDX0_BETA, IDX0_Y_ADDR_ORIG, \
        IDX0_A_ADDR_ORIG, IDX0_M_ORIG = range(13)

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(r1, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r1)
        shr(r1, r1, 2)
        band(r1, r1, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # INCX *= 4
    assert IDX0_INCX == 0
    eidx(r4, sig=ldunifrf(r3))
    shl(r0, r3, 2)

    # X_ADDR -= 4 * INCX
    sub(null, r4, IDX0_X_ADDR, cond='pushz')
    nop(sig=ldunifrf(r3))
    sub(r0, r3, r0, cond='ifa')

    # INCY *= 4 * num_qpus
    sub(null, r4, IDX0_INCY, cond='pushz')
    nop(sig=ldunifrf(r3))
    shl(r3, r3, 2)
    shl(r0, r3, num_qpus_shift, cond='ifa')

    # Y_ADDR += 4 * INCY * qpu_num
    sub(null, r4, IDX0_Y_ADDR_ORIG, cond='pushz')
    umul24(r2, r3, r1, sig=ldunifrf(r3))
    add(r0, r3, r2, cond='ifa')

    # LDA *= 4 * num_qpus
    sub(null, r4, IDX0_LDA, cond='pushz')
    nop(sig=ldunifrf(r3))
    shl(r3, r3, 2)
    shl(r0, r3, num_qpus_shift, cond='ifa')

    # A_ADDR += 4 * LDA * qpu_num - 4
    sub(null, r4, IDX0_A_ADDR_ORIG, cond='pushz')
    umul24(r2, r3, r1, sig=ldunifrf(r3))
    sub(r2, r2, 4)
    add(r0, r3, r2, cond='ifa')

    # M = M / num_qpus - 1
    sub(null, r4, IDX0_M_ORIG, cond='pushz')
    nop(sig=ldunifrf(r3))
    shr(r0, r3, num_qpus_shift, cond='ifa')
    sub(r0, r0, 1, cond='ifa')

    # N = N / 1024 - 1
    sub(null, r4, IDX0_N, cond='pushz')
    nop(sig=ldunifrf(r3))
    shr(r0, r3, 10, cond='ifa')
    sub(r0, r0, 1, cond='ifa')

    sub(null, r4, IDX0_ALPHA, cond='pushz')
    nop(sig=ldunifrf(r3))
    mov(r0, r3, cond='ifa')

    sub(null, r4, IDX0_BETA, cond='pushz')
    nop(sig=ldunifrf(r3))
    mov(r0, r3, cond='ifa')

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as ln:

        # r4 = 4 * INCX * eidx but eidx = 0 means eidx = 16
        # r5 = X_ADDR - 4 * INCX
        eidx(r4, cond='pushz')
        add(r4, 8, 8, cond='ifa').mov(broadcast, r0)
        umul24(r4, r4, r5)
        rotate(broadcast, r0, -IDX0_X_ADDR)

        for i in range(8):
            add(tmua, r5, r4).add(broadcast, r5, r4)
        for i in range(64 - 8):
            add(tmua, r5, r4, sig=ldtmu(rf[i])).add(broadcast, r5, r4)
        for i in range(8):
            nop(sig=ldtmu(rf[64 - 8 + i]))

        # X_ADDR += 4 * INCX * 64 * 16
        eidx(r4).rotate(broadcast, r5, -IDX0_X_ADDR)
        sub(null, r4, IDX0_X_ADDR, cond='pushz')
        mov(r0, r5, cond='ifa')

        # A_ADDR = A_ADDR_ORIG
        # A_ADDR_ORIG += 4 * 64 * 16
        rotate(broadcast, r0, -IDX0_A_ADDR_ORIG)
        sub(null, r4, IDX0_A_ADDR, cond='pushz')
        mov(r0, r5, cond='ifa').mov(r3, 1)
        shl(r3, r3, 12)
        sub(null, r4, IDX0_A_ADDR_ORIG, cond='pushz')
        add(r0, r0, r3, cond='ifa')

        # Y = Y_ORIG
        rotate(broadcast, r0, -IDX0_Y_ADDR_ORIG)
        sub(null, r4, IDX0_Y_ADDR, cond='pushz')
        mov(r0, r5, cond='ifa')

        # M = M_ORIG
        rotate(broadcast, r0, -IDX0_M_ORIG)
        sub(null, r4, IDX0_M, cond='pushz')
        mov(r0, r5, cond='ifa')

        with loop as lm:

            # r3 = .0
            # r4 = 4 * eidx but eidx = 0 means eidx = 16
            # r5 = A_ADDR + 4 * LDA * qpu_num - 4
            # A_ADDR += 4 * LDA * num_qpus
            eidx(r4).rotate(broadcast, r0, -IDX0_LDA)
            mov(r3, r5).rotate(broadcast, r0, -IDX0_A_ADDR)
            add(r3, r5, r3).sub(null, r4, IDX0_A_ADDR, cond='pushz')
            mov(r0, r3, cond='ifa').mov(null, r4, cond='pushz')
            add(r4, 8, 8, cond='ifa').sub(r3, 8, 8)
            shl(r4, r4, 2)

            for i in range(8):
                add(tmua, r5, r4).add(broadcast, r5, r4)
            add(tmua, r5, r4, sig=ldtmu(r2)).add(broadcast, r5, r4)
            fmul(r1, r2, rf[0])
            for i in range(1, 64 - 8):
                add(tmua, r5, r4, sig=ldtmu(r2)).add(broadcast, r5, r4)
                fadd(r3, r3, r1).fmul(r1, r2, rf[i])
            fadd(r3, r3, r1, sig=ldtmu(r2))
            fmul(r1, r2, rf[64 - 8 + 0], sig=ldtmu(r2))
            for i in range(1, 8 - 1):
                fadd(r3, r3, r1).fmul(r1, r2, rf[64 - 8 + i], sig=ldtmu(r2))
            fadd(r3, r3, r1).fmul(r1, r2, rf[64 - 8 + 7])

            # r4 = Y_ADDR + 4 * INCY * qpu_num
            fadd(r3, r3, r1).rotate(broadcast, r0, -IDX0_Y_ADDR)
            mov(tmua, r5).mov(r4, r5)

            for rot in [1, 2, 4, 8]:
                rotate(r2, r3, rot)
                fadd(r3, r3, r2).rotate(broadcast, r0, -IDX0_ALPHA)

            eidx(r1).fmul(r3, r3, r5, sig=ldtmu(r2))
            rotate(broadcast, r0, -IDX0_BETA)
            # Y_ADDR += 4 * INCY * num_qpus
            sub(null, r1, IDX0_Y_ADDR, cond='pushz').fmul(r2, r2, r5)
            fadd(r3, r3, r2).rotate(broadcast, r0, -IDX0_INCY)
            mov(tmud, r3).add(r0, r0, r5, cond='ifa')
            mov(tmua, r4).rotate(broadcast, r0, -IDX0_M, cond='pushz')
            lm.b(cond='na0')
            sub(null, r1, IDX0_M, cond='pushz')
            sub(r0, r5, 1, cond='ifa')
            nop()

        sub(null, r1, IDX0_BETA, cond='pushz')
        mov(r0, 1., cond='ifa')

        rotate(broadcast, r0, -IDX0_N, cond='pushz')
        ln.b(cond='na0')
        sub(null, r1, IDX0_N, cond='pushz')
        sub(r0, r5, 1, cond='ifa')
        nop()

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

    num_qpus, code_offset = map(int, sys.argv[1:])

    for insn in assemble(qpu_sgemv_n, num_qpus=num_qpus,
                         code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
