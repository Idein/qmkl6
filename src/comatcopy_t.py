
import sys

from videocore6 import pack_unpack
from videocore6.assembler import assemble, qpu


def ilog2(n):
    '''
    >>> ilog2(1.)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(-1)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(0)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(1)
    0
    >>> ilog2(2)
    1
    >>> ilog2(4)
    2
    >>> ilog2(8)
    3
    '''

    assert isinstance(n, int)
    assert n > 0
    assert n & (n - 1) == 0

    return n.bit_length() - 1


@qpu
def qpu_comatcopy_t(asm, *, num_qpus, tile_rows, tile_cols, subtile_rows,
                    subtile_cols, code_offset, align_cond=lambda pos: True):

    tile_size = tile_rows * tile_cols
    assert 1 <= tile_size <= 32

    assert subtile_rows * subtile_cols == 16

    IDX4_ROWS, IDX4_COLS, IDX4_ALPHA_R, IDX4_ALPHA_I, IDX4_A_I, IDX4_A_J, \
        IDX4_B_I, IDX4_B_J, IDX4_LDA, IDX4_LDB, IDX4_I, IDX4_J = range(12)

    # r0 = eidx
    # r1 = qpu_num
    eidx(r0).mov(r4, 0)
    if num_qpus == 1:
        mov(r1, 0)
    elif num_qpus == 8:
        tidx(r1)
        shr(r1, r1, 2)
        band(r1, r1, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    for idx in [IDX4_ROWS, IDX4_COLS, IDX4_ALPHA_R, IDX4_ALPHA_I, IDX4_A_I,
                IDX4_LDA, IDX4_B_I, IDX4_LDB]:
        nop(sig=ldunif)
        sub(null, r0, idx, cond='pushz')
        mov(r4, r5, cond='ifa')

    # I = ROWS / subtile_rows / tile_rows - qpu_num - num_qpus - 1
    sub(null, r0, IDX4_I, cond='pushz')
    rotate(broadcast, r4, -IDX4_ROWS)
    shr(r5, r5, ilog2(tile_rows * subtile_rows))
    sub(r5, r5, r1)
    sub(r4, r5, num_qpus + 1, cond='ifa')

    # A_I += LDA * qpu_num * tile_rows * subtile_rows * 8
    # B_I += qpu_num * tile_rows * subtile_rows * 8
    shl(r1, r1, ilog2(tile_rows * subtile_rows))
    rotate(broadcast, r4, -IDX4_LDA)
    sub(null, r0, IDX4_A_I, cond='pushz').umul24(r5, r1, r5)
    add(r4, r4, r5, cond='ifa').umul24(r5, r1, 8)
    sub(null, r0, IDX4_B_I, cond='pushz')
    add(r4, r4, r5, cond='ifa')

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as li:

        # J = COLS / subtile_cols / tile_cols - 1
        eidx(r0).rotate(broadcast, r4, -IDX4_COLS)
        sub(null, r0, IDX4_J, cond='pushz')
        shr(r5, r5, ilog2(tile_cols * subtile_cols))
        sub(r4, r5, 1, cond='ifa')

        # A_J = A_I
        # B_J = B_I
        sub(null, r0, IDX4_A_J, cond='pushz')
        rotate(broadcast, r4, -IDX4_A_I)
        mov(r4, r5, cond='ifa')
        sub(null, r0, IDX4_B_J, cond='pushz')
        rotate(broadcast, r4, -IDX4_B_I)
        mov(r4, r5, cond='ifa')

        with loop as lj:

            # r0 = A_J + eidx / subtile_cols * LDA * 4 + eidx % subtile_cols * 8
            # r1 = subtile_cols * 8
            eidx(r1).rotate(broadcast, r4, -IDX4_A_J)
            band(r0, r1, subtile_cols - 1)
            shl(r0, r0, ilog2(8))
            add(r0, r0, r5).rotate(broadcast, r4, -IDX4_LDA)
            shr(r1, r1, ilog2(subtile_cols))
            mov(r5, 1).umul24(r1, r1, r5)
            shl(r1, r5, ilog2(subtile_cols * 8)).add(r0, r0, r1)

            tmu_unroll = min(tile_size, 4)

            for i in range(tile_size + tmu_unroll):
                if i > 0 and i % tile_cols == 0:
                    # r0 = -tile_cols * subtile_cols * 8 + LDA * subtile_rows * 8
                    shl(r5, r1, ilog2(tile_cols))
                    sub(r0, r0, r5).rotate(broadcast, r4, -IDX4_LDA)
                    shl(r5, r5, ilog2(subtile_rows))
                    add(r0, r0, r5)
                if i < tmu_unroll:
                    mov(tmua, r0).mov(r5, 4)
                    add(tmua, r0, r5).add(r0, r0, r1)
                if tmu_unroll <= i:
                    j = i - tmu_unroll
                    nop(sig=ldtmu(rf[0 + j * 2]))
                    nop(sig=ldtmu(rf[1 + j * 2]))
                if tmu_unroll <= i < tile_size:
                    mov(tmua, r0).mov(r5, 4)
                    add(tmua, r0, r5).add(r0, r0, r1)

            # r0 = B_J + eidx % subtile_cols * LDB * 8 + eidx / subtile_cols * 8
            # r1 = subtile_rows * 8
            # r3 = ALPHA_R
            # r5 = ALPHA_I
            eidx(r1).rotate(broadcast, r4, -IDX4_B_J)
            shr(r0, r1, ilog2(subtile_cols))
            shl(r0, r0, ilog2(8))
            add(r0, r0, r5).rotate(broadcast, r4, -IDX4_LDB)
            band(r1, r1, subtile_cols - 1)
            mov(r5, 1).umul24(r1, r1, r5)
            shl(r1, r5, ilog2(subtile_rows * 8)).add(r0, r0, r1)
            rotate(broadcast, r4, -IDX4_ALPHA_R)
            mov(r3, r5).rotate(broadcast, r4, -IDX4_ALPHA_I)

            for i in range(tile_size):
                if i > 0 and i % tile_rows == 0:
                    # r0 = -tile_rows * subtile_rows * 8 + LDB * subtile_cols * 8
                    shl(r5, r1, ilog2(tile_rows))
                    sub(r0, r0, r5).rotate(broadcast, r4, -IDX4_LDB)
                    shl(r5, r5, ilog2(subtile_cols))
                    add(r0, r0, r5).rotate(broadcast, r4, -IDX4_ALPHA_R)
                    mov(r3, r5).rotate(broadcast, r4, -IDX4_ALPHA_I)
                j = i % tile_rows * tile_cols + i // tile_rows
                fmul(r1, rf[0 + j * 2], r3)
                fmul(r2, rf[1 + j * 2], r5)
                fsub(tmud, r1, r2).fmul(r1, rf[0 + j * 2], r5)
                mov(tmua, r0).fmul(r2, rf[1 + j * 2], r3)
                fadd(tmud, r1, r2).mov(r1, 1)
                shl(r1, r1, ilog2(subtile_rows * 8))
                add(tmua, r0, 4).add(r0, r0, r1)

            # A_J += tile_cols * subtile_cols * 8
            # B_J += LDB * tile_cols * subtile_cols * 8
            eidx(r0).mov(r5, 1)
            sub(null, r0, IDX4_A_J, cond='pushz')
            shl(r5, r5, ilog2(tile_cols * subtile_cols * 8))
            add(r4, r4, r5, cond='ifa').rotate(broadcast, r4, -IDX4_LDB)
            sub(null, r0, IDX4_B_J, cond='pushz')
            shl(r5, r5, ilog2(tile_cols * subtile_cols))
            add(r4, r4, r5, cond='ifa')

            rotate(null, r4, -IDX4_J, cond='pushz')
            lj.b(cond='na0')
            sub(null, r0, IDX4_J, cond='pushz')
            sub(r4, r4, 1, cond='ifa')
            nop()

        # A_I += LDA * num_qpus * tile_rows * subtile_rows * 8
        # B_I += num_qpus * tile_rows * subtile_rows * 8
        eidx(r0).rotate(broadcast, r4, -IDX4_LDA)
        mov(r1, 1)
        shl(r1, r1, ilog2(num_qpus * tile_rows * subtile_rows))
        sub(null, r0, IDX4_A_I, cond='pushz').umul24(r5, r1, r5)
        add(r4, r4, r5, cond='ifa').sub(null, r0, IDX4_B_I, cond='pushz')
        shl(r5, r1, ilog2(8))
        add(r4, r4, r5, cond='ifa')

        rotate(null, r4, -IDX4_I, cond='pushn')
        li.b(cond='na0')
        sub(null, r0, IDX4_I, cond='pushz')
        sub(r4, r4, num_qpus, cond='ifa')
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


def benchmark():

    from time import monotonic

    import numpy as np

    from videocore6.driver import Driver

    def run(drv, unif, src, dst, num_qpus, rows, cols, tile_rows, tile_cols,
            subtile_rows, subtile_cols, code_offset=0):

        code = drv.program(qpu_comatcopy_t, num_qpus=num_qpus,
                           tile_rows=tile_rows, tile_cols=tile_cols,
                           subtile_rows=subtile_rows, subtile_cols=subtile_cols,
                           code_offset=code_offset)

        src[:, :] = np.arange(src.size, dtype=src.dtype).reshape(src.shape)
        dst[:, :] = np.arange(dst.size, dtype=dst.dtype).reshape(dst.shape)

        unif[0] = rows
        unif[1] = cols
        unif[2] = pack_unpack('f', 'I', 1.)
        unif[3] = pack_unpack('f', 'I', 0.)
        unif[4] = src.addresses()[0, 0]
        unif[5] = cols * 8
        unif[6] = dst.addresses()[0, 0]
        unif[7] = rows * 8

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        print(f'{num_qpus} QPUs,',
              f'{rows} x {cols} matrix,',
              f'{tile_rows:2} x {tile_cols:2} tile,',
              f'{subtile_rows:2} x {subtile_cols:2} subtile:',
              f'{end - start} seconds,',
              f'{rows * cols * 8 / (end - start) * 1e-6} MB/s')

    rows = 8192
    cols = 8192

    with Driver(data_area_size=1100 * 1024 * 1024) as drv:

        unif = drv.alloc(8, dtype='uint32')
        src = drv.alloc((rows, cols), dtype='uint64')
        dst = drv.alloc((cols, rows), dtype='uint64')

        for num_qpus in [1, 8]:
            run(drv, unif, src, dst, num_qpus, rows, cols, 4, 8, 4, 4)

        for tile_rows in [2, 4, 8, 16]:
            tile_cols = 32 // tile_rows
            for subtile_rows in [2, 4, 8]:
                subtile_cols = 16 // subtile_rows
                run(drv, unif, src, dst, 8, rows, cols, tile_rows, tile_cols,
                    subtile_rows, subtile_cols)


def main():

    if len(sys.argv) == 1:
        benchmark()
        return

    num_qpus, tile_rows, tile_cols, subtile_rows, subtile_cols, code_offset \
        = map(int, sys.argv[1:])

    for insn in assemble(qpu_comatcopy_t, num_qpus=num_qpus,
                         tile_rows=tile_rows, tile_cols=tile_cols,
                         subtile_rows=subtile_rows, subtile_cols=subtile_cols,
                         code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
