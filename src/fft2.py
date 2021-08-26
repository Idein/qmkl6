
import sys

from videocore6.assembler import assemble, qpu

from common import ilog2


@qpu
def qpu_fft2(asm, *, num_qpus, do_unroll):

    g = globals()
    for i, name in enumerate(['j', 'k', 'l', 'm', 'n', 'x', 'x0', 'x1', 'y',
                              'y0', 'y1', 'buf', 'omega_addr', 'omega_r',
                              'omega_c']):
        g[f'reg_{name}'] = rf[i]

    nop(sig=ldunifrf(reg_n))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_y))
    nop(sig=ldunifrf(reg_buf))
    nop(sig=ldunifrf(reg_omega_addr))

    nop(sig=thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(null, r0, 0b1111, cond='pushz')
    if num_qpus == 8:
        b(R.exit, cond='na0')
        nop()
        nop()
        nop()
    else:
        raise Exception('num_qpus must be 8')

    b(R.set_unif, cond='always').unif_addr(reg_omega_addr)
    nop()
    nop()
    nop()
    L.set_unif

    shr(reg_k, reg_n, 1).mov(reg_j, 1)
    with loop as ljk:

        mov(reg_l, 0)
        with loop as ll:

            # m = j - 1
            # x0 = x + (j * l + eidx) * 8
            # x1 = x0 + (n / 2) * 8
            # y0 = y + (2 * j * l + eidx) * 8
            # y1 = y0 + j * 8
            sub(reg_m, reg_j, 1)
            eidx(r0).umul24(r1, reg_j, reg_l)
            add(r0, r0, r1)
            shl(r0, r0, ilog2(8)).add(r1, r0, r1)
            shl(r1, r1, ilog2(8)).add(reg_x0, reg_x, r0)
            add(reg_y0, reg_y, r1)
            shl(r0, reg_n, ilog2(8 // 2))
            add(reg_x1, reg_x0, r0, sig=ldunifrf(reg_omega_r))
            shl(r0, reg_j, ilog2(8))
            add(reg_y1, reg_y0, r0, sig=ldunifrf(reg_omega_c))

            if do_unroll:
                mov(r0, 1).mov(tmua, reg_x0)
                shl(r0, r0, ilog2(8 * 16)).mov(tmua, reg_x1)
                add(reg_x0, reg_x0, r0).add(tmua, reg_x0, 4)
                add(reg_x1, reg_x1, r0).add(tmua, reg_x1, 4)

            with loop as lm:

                mov(tmua, reg_x0)
                mov(tmua, reg_x1).mov(r2, 4)

                # eidx < 16 - (m + 1) ∴ m + eidx - 15 < 0
                # r5 = -1 - (rest - 1) = -rest
                eidx(r0).add(r1, reg_m, -15)
                add(null, r0, r1, cond='pushn').sub(r5, -1, reg_m)
                mov(r0, reg_y0).mov(r1, reg_y1)
                add(tmua, reg_x0, r2).rotate(reg_y0, r0, r5)
                add(tmua, reg_x1, r2).rotate(reg_y1, r1, r5)
                mov(reg_y0, r0, cond='ifa').mov(reg_y1, r1, cond='ifa')

                nop(sig=ldtmu(r0))
                nop(sig=ldtmu(r1))
                fadd(r2, r0, r1)
                fsub(r0, r0, r1).rotate(tmud, r2, r5)
                mov(tmua, reg_y0, sig=ldtmu(r1))
                nop(sig=ldtmu(r2))
                fadd(r3, r1, r2)
                fsub(r1, r1, r2).rotate(r3, r3, r5)
                mov(tmud, r3).fmul(r2, r0, reg_omega_r)
                add(tmua, reg_y0, 4)
                fmul(r3, r1, reg_omega_c)
                fsub(r2, r2, r3).add(reg_m, reg_m, -16, cond='pushn')
                rotate(tmud, r2, r5)
                mov(tmua, reg_y1).fmul(r2, r0, reg_omega_c)
                fmul(r3, r1, reg_omega_r)
                fadd(r2, r2, r3).mov(r0, 1)
                rotate(tmud, r2, r5)
                add(tmua, reg_y1, 4)

                lm.b(cond='na0')
                shl(r0, r0, ilog2(8 * 16))
                add(reg_x0, reg_x0, r0).add(reg_x1, reg_x1, r0)
                add(reg_y0, reg_y0, r0).add(reg_y1, reg_y1, r0)

            nop(sig=ldtmu(null)) if do_unroll else nop()
            add(reg_l, reg_l, 1)
            sub(null, reg_l, reg_k, cond='pushn')
            ll.b(cond='a0')
            nop(sig=ldtmu(null)) if do_unroll else nop()
            nop(sig=ldtmu(null)) if do_unroll else nop()
            nop(sig=ldtmu(null)) if do_unroll else nop()

        shr(reg_k, reg_k, 1, cond='pushz')
        ljk.b(cond='na0')
        shl(reg_j, reg_j, 1).sub(null, reg_j, 1, cond='pushz')
        mov(reg_y, reg_buf, cond='ifa').mov(r0, reg_y)
        mov(reg_y, reg_x, cond='ifna').mov(reg_x, r0)

    L.exit

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

    from cmath import rect
    from math import pi
    from struct import pack, unpack
    from time import monotonic

    from numpy.fft import fft, ifft
    from numpy.random import default_rng

    from videocore6.driver import Driver

    def run(n, drv, unif, src, dst, tmp, twiddle, num_qpus, do_unroll,
            code_offset=0):

        ref = fft(src) if is_forward else ifft(src, norm='forward')

        code = drv.program(qpu_fft2, num_qpus=num_qpus, do_unroll=do_unroll)

        unif[0] = n
        unif[1] = src.addresses()[0]
        unif[2] = tmp.addresses()[0]
        unif[3] = dst.addresses()[0]
        if is_swapped:
            unif[2], unif[3] = unif[3], unif[2]
        unif[4] = twiddle.addresses()[0]

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        err_abs = abs(dst - ref)
        err_rel = abs(err_abs / ref)

        print(', '.join([f'n = 2^{ilog2(n)}', f'do_unroll = {do_unroll}',
                         f'{end - start} seconds',
                         f'{n / (end - start) * 1e-6} Melem/s',
                         f'err_abs = [{min(err_abs)}, {max(err_abs)}]',
                         f'err_rel = [{min(err_rel)}, {max(err_rel)}]']))

    num_qpus = 8
    is_forward = True
    do_unroll = False

    for ilog2n in range(1, 23):
        n = pow(2, ilog2n)
        is_swapped = ilog2n % 2 != 0

        with Driver(data_area_size=8 * n * 4 + 4096) as drv:

            unif = drv.alloc(5, dtype='uint32')
            src = drv.alloc(n, dtype='csingle')
            dst = drv.alloc(n, dtype='csingle')
            tmp = drv.alloc(n, dtype='csingle')
            twiddle = drv.alloc(n - 1, dtype='csingle')

            rng = default_rng(0xdeadbeef)
            src[:] = rng.random(n, 'float32') + rng.random(n, 'float32') * 1j
            dst[:] = 0
            tmp[:] = 0

            c = (-1 if is_forward else 1) * pi
            j = 0
            for ilog2k in range(ilog2n - 1, -1, -1):
                k = pow(2, ilog2k)
                for l in range(k):
                    twiddle[j] = rect(1, c * l / k)
                    j += 1
            assert j == len(twiddle)

            run(n, drv, unif, src, dst, tmp, twiddle, num_qpus, do_unroll)


def main():

    if len(sys.argv) == 1:
        benchmark()
        return

    num_qpus, do_unroll = map(int, sys.argv[1:])

    for insn in assemble(qpu_fft2, num_qpus=num_qpus, do_unroll=do_unroll):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
