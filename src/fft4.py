
import sys

from videocore6.assembler import assemble, qpu

from common import ilog2


@qpu
def qpu_fft4(asm, *, num_qpus, is_forward):

    g = globals()
    for i, name in enumerate(['j', 'k', 'l', 'm', 'n', 'x_orig', 'x', 'y_orig',
                              'y', 'buf', 'omega_r', 'omega_c', 'c0r', 'c0c',
                              'c1r', 'c1c', 'c2r', 'c2c', 'c3r', 'c3c']):
        g[f'reg_{name}'] = rf[i]

    nop(sig=ldunifrf(reg_n))
    nop(sig=ldunifrf(reg_x_orig))
    nop(sig=ldunifrf(reg_y_orig))
    nop(sig=ldunifrf(reg_buf))

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

    b(R.set_unif, cond='always').unif_addr(absolute=True)
    nop()
    nop()
    nop()
    L.set_unif

    shr(reg_k, reg_n, ilog2(4))
    mov(reg_j, 1)
    with loop as ljk:

        mov(reg_l, 0)
        with loop as ll:

            # m = j - 1
            # x = x_orig + (j * l + eidx) * 8
            # y = y_orig + (4 * j * l + eidx) * 8
            sub(reg_m, reg_j, 1)
            eidx(r0).umul24(r1, reg_j, reg_l)
            add(r1, r0, r1).umul24(r2, r1, 4)
            shl(r1, r1, ilog2(8)).add(r2, r2, r0)
            shl(r2, r2, ilog2(8)).add(reg_x, reg_x_orig, r1)
            add(reg_y, reg_y_orig, r2)

            with loop as lm:

                # r0 = n / 4 * 8 = 2 n, r1 = 2 n + 4
                mov(tmua, reg_x).add(r0, reg_n, reg_n)
                add(tmua, reg_x, 4).add(r1, r0, 4)

                # r5 = 4 n, r1 = 4 n + 8
                add(tmua, reg_x, r0).add(r5, r0, r0)
                add(tmua, reg_x, r1).add(r1, r1, r1)

                # r0 = r0 + r5 = 2 n + 4 n = 6 n, r1 = r1 - 4 = 4 n + 4
                add(tmua, reg_x, r5).sub(r1, r1, 4)
                add(tmua, reg_x, r1).add(r0, r0, r5)

                # r0 = r0 + 4 = 6 n + 4
                add(tmua, reg_x, r0).add(r0, r0, 4)
                add(tmua, reg_x, r0)

                # eidx < 16 - (m + 1) ∴ m + eidx - 15 < 0
                # r5 = -1 - (rest - 1) = -rest
                eidx(r0).add(r1, reg_m, -15)
                add(null, r0, r1, cond='pushn').sub(r5, -1, reg_m)
                mov(r0, reg_y)
                rotate(reg_y, r0, r5)
                mov(reg_y, r0, cond='ifa')

                nop(sig=ldtmu(reg_c0r))
                nop(sig=ldtmu(reg_c0c))
                nop(sig=ldtmu(reg_c1r))
                nop(sig=ldtmu(reg_c1c))
                nop(sig=ldtmu(reg_c2r))
                nop(sig=ldtmu(reg_c2c))
                nop(sig=ldtmu(reg_c3r))
                nop(sig=ldtmu(reg_c3c))

                # d0, d1, d2, d3 are stored to c0, c2, c1, c3, resp.
                fadd(r0, reg_c0r, reg_c2r)
                fadd(r1, reg_c0c, reg_c2c)
                fsub(reg_c2r, reg_c0r, reg_c2r).mov(reg_c0r, r0)
                fsub(reg_c2c, reg_c0c, reg_c2c).mov(reg_c0c, r1)
                fadd(r0, reg_c1r, reg_c3r)
                fadd(r1, reg_c1c, reg_c3c)
                if is_forward:
                    fsub(r1, reg_c1c, reg_c3c).mov(reg_c1c, r1)
                    fsub(r0, reg_c3r, reg_c1r).mov(reg_c1r, r0)
                else:
                    fsub(r1, reg_c3c, reg_c1c).mov(reg_c1c, r1)
                    fsub(r0, reg_c1r, reg_c3r).mov(reg_c1r, r0)
                mov(reg_c3r, r1)
                mov(reg_c3c, r0)

                reg_d0r = reg_c0r
                reg_d0c = reg_c0c
                reg_d1r = reg_c2r
                reg_d1c = reg_c2c
                reg_d2r = reg_c1r
                reg_d2c = reg_c1c
                reg_d3r = reg_c3r
                reg_d3c = reg_c3c

                fadd(r0, reg_d0r, reg_d2r)
                rotate(tmud, r0, r5)
                fadd(r0, reg_d0c, reg_d2c)
                mov(tmua, reg_y)
                rotate(tmud, r0, r5)
                add(tmua, reg_y, 4)
                shl(r0, reg_j, ilog2(8))
                add(reg_y, reg_y, r0)

                fadd(r0, reg_d1r, reg_d3r, sig=ldunifrf(reg_omega_r))
                fadd(r1, reg_d1c, reg_d3c, sig=ldunifrf(reg_omega_c))
                fmul(r2, r0, reg_omega_r)
                fmul(r3, r1, reg_omega_c)
                fsub(r2, r2, r3)
                rotate(tmud, r2, r5)
                mov(tmua, reg_y).fmul(r2, r0, reg_omega_c)
                fmul(r3, r1, reg_omega_r)
                fadd(r2, r2, r3)
                rotate(tmud, r2, r5)
                add(tmua, reg_y, 4)
                shl(r0, reg_j, ilog2(8))
                add(reg_y, reg_y, r0)

                fsub(r0, reg_d0r, reg_d2r, sig=ldunifrf(reg_omega_r))
                fsub(r1, reg_d0c, reg_d2c, sig=ldunifrf(reg_omega_c))
                fmul(r2, r0, reg_omega_r)
                fmul(r3, r1, reg_omega_c)
                fsub(r2, r2, r3)
                rotate(tmud, r2, r5)
                mov(tmua, reg_y).fmul(r2, r0, reg_omega_c)
                fmul(r3, r1, reg_omega_r)
                fadd(r2, r2, r3)
                rotate(tmud, r2, r5)
                add(tmua, reg_y, 4)
                shl(r0, reg_j, ilog2(8))
                add(reg_y, reg_y, r0)

                fsub(r0, reg_d1r, reg_d3r, sig=ldunifrf(reg_omega_r))
                fsub(r1, reg_d1c, reg_d3c, sig=ldunifrf(reg_omega_c))
                fmul(r2, r0, reg_omega_r)
                fmul(r3, r1, reg_omega_c)
                fsub(r2, r2, r3)
                rotate(tmud, r2, r5)
                mov(tmua, reg_y).fmul(r2, r0, reg_omega_c)
                fmul(r3, r1, reg_omega_r)
                fadd(r2, r2, r3).add(reg_m, reg_m, -16, cond='pushn')
                rotate(tmud, r2, r5)
                add(tmua, reg_y, 4)
                add(r0, 12, 12)  # 8 * 3
                mov(r1, 1).umul24(r0, reg_j, r0)

                lm.b(cond='na0').unif_addr(absolute=False)
                sub(reg_y, reg_y, r0)
                shl(r0, r1, ilog2(8 * 16))
                add(reg_x, reg_x, r0).add(reg_y, reg_y, r0)

            add(reg_l, reg_l, 1)
            sub(null, reg_l, reg_k, cond='pushn')
            ll.b(cond='a0')
            nop()
            nop()
            nop()

        shr(reg_k, reg_k, ilog2(4), cond='pushz')
        ljk.b(cond='na0')
        shl(reg_j, reg_j, ilog2(4)).sub(null, reg_j, 2, cond='pushn')
        mov(reg_y_orig, reg_buf, cond='ifa').mov(r0, reg_y_orig)
        mov(reg_y_orig, reg_x_orig, cond='ifna').mov(reg_x_orig, r0)

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

    def run(n, is_forward, drv, unif, src, dst, tmp, twiddle, num_qpus,
            code_offset=0):

        ref = fft(src) if is_forward else ifft(src, norm='forward')

        code = drv.program(qpu_fft4, num_qpus=num_qpus, is_forward=is_forward)

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

        print(', '.join([f'n = 2^{ilog2(n)}', f'{end - start} seconds',
                         f'{n / (end - start) * 1e-6} Melem/s',
                         f'err_abs = [{min(err_abs)}, {max(err_abs)}]',
                         f'err_rel = [{min(err_rel)}, {max(err_rel)}]']))

    num_qpus = 8
    is_forward = True

    for ilog4n in range(1, 12):
        n = pow(4, ilog4n)
        is_swapped = ilog4n % 2 != 0

        with Driver(data_area_size=8 * n * 5 + 4096) as drv:

            unif = drv.alloc(5, dtype='uint32')
            src = drv.alloc(n, dtype='csingle')
            dst = drv.alloc(n, dtype='csingle')
            tmp = drv.alloc(n, dtype='csingle')
            twiddle = drv.alloc((n - 1) // 3 * 7, dtype='float32')

            rng = default_rng(0xdeadbeef)
            src[:] = rng.random(n, 'float32') + rng.random(n, 'float32') * 1j
            dst[:] = 0
            tmp[:] = 0

            c = pi / (-2 if is_forward else 2)
            j = 0
            for ilog4k in range(ilog4n - 1, -1, -1):
                k = pow(4, ilog4k)
                for l in range(k):
                    for m in range(1, 4):
                        omega = rect(1, c * l * m / k)
                        twiddle[j], twiddle[j + 1] = omega.real, omega.imag
                        j += 2
                    twiddle[j] = unpack('f', pack('i', 4 * -7))[0]
                    j += 1
            assert j == len(twiddle)

            run(n, is_forward, drv, unif, src, dst, tmp, twiddle, num_qpus)


def main():

    if len(sys.argv) == 1:
        benchmark()
        return

    num_qpus, is_forward = map(int, sys.argv[1:])

    for insn in assemble(qpu_fft4, num_qpus=num_qpus, is_forward=is_forward):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
