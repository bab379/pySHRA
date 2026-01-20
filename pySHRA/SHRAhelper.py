import numpy as np
import sympy as sp
from susceptibility_tensors import getX2TensorLab, getX3TensorLab
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


class Constants:
    c = 3e8
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

class RotationMatrices:
    def __init__(self):
        phi = sp.Symbol('phi',real=True)
        self.Q_z = np.array([[sp.cos(phi), -sp.sin(phi), 0], [sp.sin(phi), sp.cos(phi), 0], [0, 0, 1]])
        self.phi = phi
        self.pi = sp.pi

constants = Constants()

def rootPicker(vz):
    """
    Enforce the physical branch of sqrt for wavevectors:
      - Im(vz) >= 0 ensures decaying (evanescent) or upward propagation.
      - If purely real, keep Re(vz) >= 0 (upward-propagating convention).
    """
    # flip sign if Im(vz) < 0
    vz = np.where(np.imag(vz) < 0, -vz, vz)
    # for purely real vz, make Re(vz) >= 0
    vz = np.where(np.isclose(np.imag(vz), 0) & (np.real(vz) < 0), -vz, vz)
    return vz.item() if np.ndim(vz) == 0 else vz

def calc_v_z_iso(eps,v0,kappa_mag):
    v_z = np.sqrt(eps * (v0 ** 2) - (kappa_mag ** 2), dtype=complex)
    v_z = rootPicker(v_z)
    return v_z

class IsotropicInterface:
    def calc_v_z_iso(self, eps, v0, kappa_mag):
        v_z = np.sqrt(eps * (v0 ** 2) - (kappa_mag ** 2), dtype=complex)
        v_z = rootPicker(v_z)
        return v_z

    def calc_fresnel_mats(self,epsI,epsII,v_z_I,v_z_II):
        r_ss = (v_z_I - v_z_II) / (v_z_I + v_z_II)
        t_ss = (2 * v_z_I) / (v_z_I + v_z_II)
        r_pp = (epsII * v_z_I - epsI * v_z_II) / (epsII * v_z_I + epsI * v_z_II)
        t_pp = (2 * v_z_I * np.sqrt(epsI, dtype=complex) * np.sqrt(epsII, dtype=complex)) / (epsII * v_z_I + epsI * v_z_II)
        r_sp = 0.0
        r_ps = 0.0
        t_sp = 0.0
        t_ps = 0.0
        R = np.array([[r_ss, r_sp], [r_ps, r_pp]])
        # R = 0*np.array([[r_ss, r_sp], [r_ps, r_pp]])
        # input('testing reflectance = 0')
        T = np.array([[t_ss, t_sp], [t_ps, t_pp]])
        print('R: ', R)
        print('T: ', T)
        return R,T

    def calc_d(self,s,kappa,v_z_II,eps_II,b,z,v0):
        # matrix containing s,p basis vectors of isotropic medium II
        kappa_mag = np.linalg.norm(kappa)
        kappa_hat = kappa/kappa_mag
        p_II = (kappa_mag*z+v_z_II*kappa_hat)/(np.sqrt(eps_II,dtype=complex)*v0)
        p_I = b[:,1]
        print('p_I',p_I)
        print('p_II',p_II)
        d = np.column_stack((s,p_II))
        return d

class AnisotropicInterfaceAnalytical:
    # wavevector components, field components, and fresnel matrices
    # computed using Lekner's formulae given in J. Phys.: Condens. Matter 3 (1991) 6121-6133
    # Note that Lekner chooses z_lekner = -z and y_lekner = -y
    # Thus we must flip the sign on the angle cosine between c and z
    def __init__(self,angles_xtal_to_lab,nCmp,phi,nPhi=1000):
        angles_xtal_to_lekner = [angles_xtal_to_lab[0],angles_xtal_to_lab[1]+sp.pi,angles_xtal_to_lab[2]+sp.pi]
        angle_cosines_lekner = [sp.cos(alpha) for alpha in angles_xtal_to_lekner]
        self.phi = phi
        self.phi_vals = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
        angle_cosines_funcs = [sp.lambdify(self.phi, alpha, 'numpy') for alpha in angle_cosines_lekner]
        angle_cosines_eval = [f(self.phi_vals) for f in angle_cosines_funcs]
        self.alpha1 = angle_cosines_eval[0]
        self.alpha2 = angle_cosines_eval[1]
        self.alpha3 = angle_cosines_eval[2]
        self.nCmp = nCmp
        self.precompute_fourier_matrix(nCmp=self.nCmp)


    def rootPickerLekner(self,vz):
        return rootPicker(vz)

    def calc_v_z_iso(self, eps, v0, kappa_mag):
        v_z = np.sqrt(eps * (v0 ** 2) - (kappa_mag ** 2), dtype=complex)
        v_z = self.rootPickerLekner(v_z)
        return v_z

    def calc_v_z_ordinary(self,eps,v0,kappa_mag):
        eps_o = eps[0]
        v_z = np.sqrt(eps_o * (v0 ** 2) - (kappa_mag ** 2), dtype=complex)
        v_z = self.rootPickerLekner(v_z)
        return v_z

    def pick_vz_forward(self,vz_plus, vz_minus):
        """
        Vectorized version of pick_vz_forward:
        Selects the physically forward-propagating root elementwise.
          - Prefer Im(vz) >= 0 (decaying or upward-propagating)
          - If purely real, pick Re(vz) >= 0
        """
        vz_plus = np.asarray(vz_plus)
        vz_minus = np.asarray(vz_minus)

        # 1. Prefer Im(vz_plus) >= 0
        cond1 = np.imag(vz_plus) >= 0
        # 2. Otherwise, if Im(vz_minus) >= 0, take vz_minus
        cond2 = np.imag(vz_minus) >= 0
        # 3. Otherwise, fallback for purely real: pick positive Re(vz_plus)
        cond3 = np.isclose(np.imag(vz_plus), 0) & (np.real(vz_plus) >= 0)

        # Combine conditions in priority order
        vz = np.where(cond1, vz_plus,
                      np.where(cond2, vz_minus,
                               np.where(cond3, vz_plus, vz_minus)))

        # Keep scalar output if inputs are scalar
        return vz.item() if vz.shape == () else vz

    def calc_v_z_extraordinary(self,eps,v0,kappa_mag):
        eps_o, eps_e = eps[0], eps[1]
        delta_eps = eps_e - eps_o
        term1 = eps_o + (self.alpha3**2)*delta_eps
        term2 = eps_e - (self.alpha2**2)*delta_eps
        d_root = np.sqrt(eps_o*(eps_e*term1*(v0**2)-term2*(kappa_mag**2)),dtype=complex)
        v_z_plus = (d_root - self.alpha1*self.alpha3*kappa_mag*delta_eps)/term1
        v_z_minus = (-1*d_root - self.alpha1 * self.alpha3 * kappa_mag * delta_eps) / term1
        v_z = self.pick_vz_forward(v_z_plus, v_z_minus)
        return v_z

    def calc_uniaxial_fields(self,eps,v_z_o,v_z_e,v0,kappa_mag):
        eps_o, eps_e = eps[0], eps[1]
        E_ordinary = np.array([-self.alpha2*v_z_o,
                               self.alpha1*v_z_o-self.alpha3*kappa_mag,
                               self.alpha2*kappa_mag],dtype=complex)
        E_extraordinary = np.array([self.alpha1*(v_z_o**2)-self.alpha3*v_z_e*kappa_mag,
                                    self.alpha2*eps_o*(v0**2),
                                    self.alpha3*(eps_o*(v0**2)-(v_z_e**2))-self.alpha1*v_z_e*kappa_mag],dtype=complex)
        N_o = 1/np.linalg.norm(E_ordinary,axis=0)
        N_e = 1/np.linalg.norm(E_extraordinary,axis=0)

        E_o_unit = N_o*E_ordinary
        E_e_unit = N_e*E_extraordinary
        return E_o_unit, E_e_unit

    def rm_sing(self,arr):
        # plt.plot(self.phi_vals, arr)
        arr = arr.copy()
        bad_phis = np.deg2rad([0, 90, 180, 270, 360]) % (2 * np.pi)
        tol = 1e-6  # radians — small tolerance for floating-point comparisons

        # Create interpolation function from all other points
        good_mask = np.ones_like(self.phi_vals, dtype=bool)
        for bp in bad_phis:
            good_mask[np.isclose(self.phi_vals, bp, atol=tol)] = False

        f = interp1d(self.phi_vals[good_mask], arr[good_mask], kind='cubic', fill_value='extrapolate')

        # Replace only the bad indices
        for bp in bad_phis:
            idx = np.where(np.isclose(self.phi_vals, bp, atol=tol))[0]
            if len(idx) > 0:
                arr[idx] = f(self.phi_vals[idx])
        # plt.plot(self.phi_vals,arr)
        # plt.show()
        return arr

    def calc_fresnel_mats(self,E_o_unit,E_e_unit,kappa_mag,v0,v_z_o,v_z_e,theta,eps_iso,medIII_data=[],medII_thickness=np.inf):
        v_z_iso = self.calc_v_z_iso(eps_iso,v0,kappa_mag)
        E_o_x, E_o_y, E_o_z = E_o_unit[0], E_o_unit[1], E_o_unit[2]
        E_e_x, E_e_y, E_e_z = E_e_unit[0], E_e_unit[1], E_e_unit[2]
        if len(medIII_data)==0:
            A = (v_z_o + v_z_iso + kappa_mag*np.tan(theta))*E_o_x-kappa_mag*E_o_z
            B = (v_z_e + v_z_iso + kappa_mag*np.tan(theta))*E_e_x-kappa_mag*E_e_z
            D = (v_z_iso + v_z_e)*A*E_e_y - (v_z_iso + v_z_o)*B*E_o_y
            q_t = eps_iso*(v0**2)/v_z_iso
            n_iso = np.sqrt(eps_iso,dtype=complex)

            r_ss = ((v_z_iso - v_z_e)*A*E_e_y - (v_z_iso - v_z_o)*B*E_o_y)/D
            r_ps = (2*n_iso*v0*(A*E_e_x - B*E_o_x))/D
            t_os = (-2*v_z_iso*B)/D
            t_es = (2*v_z_iso*A)/D # notice the + sign (lekner appears to have made a typo..)


            r_pp = (2*q_t/D)*((v_z_iso + v_z_e)*E_o_x*E_e_y - (v_z_iso + v_z_o)*E_e_x*E_o_y)-1
            r_sp = (2*n_iso*v0*(v_z_e - v_z_o)*E_o_y*E_e_y)/D
            t_op = (2*n_iso*v0*(v_z_iso + v_z_e)*E_e_y)/D
            t_ep = (-2*n_iso*v0*(v_z_iso + v_z_o)*E_o_y)/D
            medIII_sym_results = []


        else:
            [v_z_o_III, v_z_e_III, E_o_unit_III, E_e_unit_III] = medIII_data
            # symbols = [
            #     'd', 'K', 'theta', 'q1', 'qo2', 'qe2', 'qo3', 'qe3',
            #     'Eox2', 'Eoy2', 'Eoz2', 'Eex2', 'Eey2', 'Eez2',
            #     'Eox3', 'Eoy3', 'Eoz3', 'Eex3', 'Eey3', 'Eez3'
            # ]
            S = np.diag([1, -1, -1])
            E_o_unit = S @ E_o_unit
            E_e_unit = S @ E_e_unit
            E_o_unit_III = S @ E_o_unit_III
            E_e_unit_III = S @ E_e_unit_III
            vals = [medII_thickness*1e-9,kappa_mag,theta,v_z_iso,v_z_o,v_z_e,v_z_o_III,v_z_e_III]
            vals = vals + [E_o_unit[0],E_o_unit[1],E_o_unit[2],E_e_unit[0],E_e_unit[1],E_e_unit[2]]
            vals = vals + [E_o_unit_III[0],E_o_unit_III[1],E_o_unit_III[2]]
            vals = vals + [E_e_unit_III[0], E_e_unit_III[1], E_e_unit_III[2]]
            print(self.sol_iso_uni_uni_dict.keys())
            r_ss = self.rm_sing(self.sol_iso_uni_uni_dict['rss12'](*vals))
            r_ps = self.rm_sing(self.sol_iso_uni_uni_dict['rps12'](*vals))
            t_os = self.rm_sing(self.sol_iso_uni_uni_dict['tos12'](*vals))
            t_es = self.rm_sing(self.sol_iso_uni_uni_dict['tes12'](*vals))
            r_pp = self.rm_sing(self.sol_iso_uni_uni_dict['rpp12'](*vals))
            r_sp = self.rm_sing(self.sol_iso_uni_uni_dict['rsp12'](*vals))
            t_op = self.rm_sing(self.sol_iso_uni_uni_dict['top12'](*vals))
            t_ep = self.rm_sing(self.sol_iso_uni_uni_dict['tep12'](*vals))

            # jones matrix elements of the second interface
            r_os23 = self.rm_sing(self.sol_iso_uni_uni_dict['ros23'](*vals))
            r_es23 = self.rm_sing(self.sol_iso_uni_uni_dict['res23'](*vals))
            r_op23 = self.rm_sing(self.sol_iso_uni_uni_dict['rop23'](*vals))
            r_ep23 = self.rm_sing(self.sol_iso_uni_uni_dict['rep23'](*vals))
            t_os23 = self.rm_sing(self.sol_iso_uni_uni_dict['tos23'](*vals))
            t_es23 = self.rm_sing(self.sol_iso_uni_uni_dict['tes23'](*vals))
            t_op23 = self.rm_sing(self.sol_iso_uni_uni_dict['top23'](*vals))
            t_ep23 = self.rm_sing(self.sol_iso_uni_uni_dict['tep23'](*vals))

        R = np.array([[r_ss, r_sp], [r_ps, r_pp]])
        T = np.array([[t_os, t_op], [t_es, t_ep]])
        R_sym = sp.Matrix.zeros(2,2)
        T_sym = sp.Matrix.zeros(2,2)
        R_rrmse = sp.Matrix.zeros(2, 2)
        T_rrmse = sp.Matrix.zeros(2, 2)
        E_o_unit_sym = sp.Matrix.zeros(3,1)
        E_e_unit_sym = sp.Matrix.zeros(3,1)
        for i in range(0,R.shape[0]):
            for j in range(0,R.shape[1]):
                recon_sym, rrmse = self.get_fourier_cmps(R[i,j,:],'R_'+str(i)+str(j))
                R_sym[i,j] = recon_sym
                R_rrmse[i,j] = rrmse
        for i in range(0,T.shape[0]):
            for j in range(0,T.shape[1]):
                recon_sym, rrmse = self.get_fourier_cmps(T[i,j,:],'T'+str(i)+str(j))
                T_sym[i,j] = recon_sym
                T_rrmse[i,j] = rrmse
        for i in range(0,len(E_o_unit)):
            recon_sym, rrmse = self.get_fourier_cmps(E_o_unit[i],'E_o'+str(i))
            E_o_unit_sym[i] = recon_sym
            E_o_unit_rrmse = rrmse
        for i in range(0,len(E_e_unit)):
            recon_sym, rrmse = self.get_fourier_cmps(E_e_unit[i],'E_e'+str(i))
            E_e_unit_sym[i] = recon_sym
            E_e_unit_rrmse = rrmse

        recon_sym, rrmse = self.get_fourier_cmps(v_z_e,'v_z_e')
        v_z_e_sym = recon_sym

        if len(medIII_data) > 0:
            R_23 = np.array([[r_os23, r_op23], [r_es23, r_ep23]])
            T_23 = np.array([[t_os23, t_op23], [t_es23, t_ep23]])
            R_23_sym = sp.Matrix.zeros(2, 2)
            T_23_sym = sp.Matrix.zeros(2, 2)
            E_o_unit_III_sym = sp.Matrix.zeros(3, 1)
            E_e_unit_III_sym = sp.Matrix.zeros(3, 1)
            for i in range(0, R_23.shape[0]):
                for j in range(0, R_23.shape[1]):
                    recon_sym, rrmse = self.get_fourier_cmps(R_23[i, j, :], 'R_23_' + str(i) + str(j))
                    R_23_sym[i, j] = recon_sym
            for i in range(0, T_23.shape[0]):
                for j in range(0, T_23.shape[1]):
                    recon_sym, rrmse = self.get_fourier_cmps(T_23[i, j, :], 'T_23' + str(i) + str(j))
                    T_23_sym[i, j] = recon_sym
            for i in range(0, len(E_o_unit_III)):
                recon_sym, rrmse = self.get_fourier_cmps(E_o_unit_III[i], 'E_o_III' + str(i))
                E_o_unit_III_sym[i] = recon_sym
            for i in range(0, len(E_e_unit_III)):
                recon_sym, rrmse = self.get_fourier_cmps(E_e_unit_III[i], 'E_e_III' + str(i))
                E_e_unit_III_sym[i] = recon_sym
            recon_sym, rrmse = self.get_fourier_cmps(v_z_e_III, 'v_z_e_III')
            v_z_e_III_sym = recon_sym
            medIII_sym_results = [R_23_sym,T_23_sym,E_o_unit_III_sym,E_e_unit_III_sym,v_z_e_III_sym]


        return R_sym,T_sym,E_o_unit_sym,E_e_unit_sym,v_z_e_sym,medIII_sym_results

    def precompute_fourier_matrix(self, nCmp, nPhi=1000):
        self.phi_vals = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
        n_array = np.arange(-nCmp, nCmp + 1).reshape(-1, 1)  # includes negative freqs
        self.fourier_matrix = np.exp(-1j * n_array * self.phi_vals)  # (2nCmp+1, nPhi)

    def get_fourier_cmps(self,m_ij,label='label'):
        coef_list = (self.fourier_matrix @ m_ij) / float(len(self.phi_vals))
        iso, cos_n, sin_n = extract_cos_sin_coeffs(coef_list)
        recon = reconstruct_from_cmp(iso, cos_n, sin_n, self.phi_vals)
        num = np.mean(np.abs(m_ij - recon) ** 2)
        den = np.mean(np.abs(m_ij) ** 2)
        rrmse = float(np.sqrt(num) / (np.sqrt(den) + 1e-30))
        if np.abs(rrmse) > 0.05 and np.max(np.abs(m_ij)) > 1e-8:
        #if True:
            print('max(abs(m_ij))',np.max(np.abs(m_ij)))
            print('Increase number of fourier components to improve accuracy...')
            plt.plot(self.phi_vals,np.real(m_ij),color='black')
            plt.plot(self.phi_vals,np.real(recon),color='red')
            plt.title('Reconstruction of: ' + label)
            plt.show()

        recon_sym = reconstruct_from_cmp_sym(iso,cos_n,sin_n,self.phi)
        return recon_sym, rrmse

    def convert_bases(self, kappa, v_m, v_p, b, f, axes, R_sym, T_sym,
                      E_o_unit_sym, E_e_unit_sym):
        S = sp.diag(1, -1, -1)
        # Use SymPy vectors
        v_m = sp.Matrix(v_m)
        v_p = sp.Matrix(v_p)
        kappa = sp.Matrix(kappa)
        v_m_lekner = S * v_m
        v_p_lekner = S * v_p
        kappa_lekner = S * kappa
        z_lekner = sp.Matrix([0,0,1]) # lekner z vector in *lekner* coordinates
        s_lekner = z_lekner.cross(kappa_lekner.normalized())
        p_m_lekner = s_lekner.cross(v_m_lekner.normalized())
        p_p_lekner = v_p_lekner.normalized().cross(s_lekner)


        # Build symbolic basis matrices
        b_lekner = sp.Matrix.hstack(s_lekner, p_m_lekner)
        f_lekner = sp.Matrix.hstack(s_lekner, p_p_lekner)
        d_lekner = sp.Matrix.hstack(E_o_unit_sym, E_e_unit_sym)
        b_lekner_inv = b_lekner.pinv()

        f_sym = sp.Matrix(f)
        b_sym = sp.Matrix(b)
        f_inv = f_sym.pinv()
        d_sym = S * d_lekner

        T = T_sym * b_lekner_inv * S * b_sym
        R = f_inv * S * f_lekner * R_sym * b_lekner_inv * S * b_sym

        return R, T, d_sym


    def prune_small_terms_matrix(self,M, tol=1e-8):
        M = sp.Matrix(M)
        M_clean = sp.zeros(*M.shape)
        for i in range(M.rows):
            for j in range(M.cols):
                expr = sp.expand(M[i, j])
                new_terms = []
                for term in expr.as_ordered_terms():
                    coeff, func = term.as_coeff_Mul()
                    try:
                        mag = abs(float(coeff))
                    except Exception:
                        mag = abs(coeff.evalf())
                    if mag >= tol:
                        new_terms.append(term)
                M_clean[i, j] = sum(new_terms)
        return M_clean


def calc_v0(omega):
    return omega/constants.c


class mediumI:
    def __init__(self,theta,point_group,rotation_matrix):
        self.x = np.array(constants.axes[0])
        self.y = np.array(constants.axes[1])
        self.z = np.array(constants.axes[2])
        self.v_fmHat = np.array([np.sin(theta),0,-np.cos(theta)]) # unit wavevector of incident wave
        self.point_group = point_group
        self.rotation_matrix = rotation_matrix
        self.X3_lab, self.X3_xtal = getX3TensorLab(self.point_group,self.rotation_matrix)

    def setup(self,omega_f,eps_f,eps_SH):
        self.eps_f = eps_f
        self.eps_SH = eps_SH
        self.epsDict = {'fund':self.eps_f,'SH':self.eps_SH}
        self.omega_f = omega_f
        self.omega_SH = 2*self.omega_f
        self.v0_f = calc_v0(self.omega_f)
        self.v0_SH = calc_v0(self.omega_SH)
        self.v_fMag = self.v0_f*np.sqrt(eps_f,dtype=complex)
        self.v_SHMag = self.v0_SH*np.sqrt(eps_SH,dtype=complex)
        self.kappa_f = np.abs(self.v_fMag)*np.array([self.v_fmHat[0],self.v_fmHat[1],0])
        self.kappa_fMag = np.linalg.norm(self.kappa_f)
        self.kappaHat = self.kappa_f / self.kappa_fMag

        self.v_f_z = calc_v_z_iso(eps=self.eps_f,v0=self.v0_f,kappa_mag=self.kappa_fMag)
        self.v_fm = self.kappa_f - self.v_f_z*self.z
        self.v_fp = self.kappa_f + self.v_f_z*self.z

        self.kappa_SHMag = 2*self.kappa_fMag
        self.kappa_SH = self.kappa_SHMag*self.kappaHat
        self.v_SH_z = calc_v_z_iso(eps=self.eps_SH, v0=self.v0_SH, kappa_mag=self.kappa_SHMag)
        self.v_SHm = self.kappa_SH - self.v_SH_z*self.z
        self.v_SHp = self.kappa_SH + self.v_SH_z*self.z
        self.vDict = {'fund':{'minus':self.v_fm,'plus':self.v_fp},'SH':{'minus':self.v_SHm,'plus':self.v_SHp}}
        self.vCmpDict = {'fund':{'kappa_mag':self.kappa_fMag,'v_z':self.v_f_z},'SH':{'kappa_mag':self.kappa_SHMag,'v_z':self.v_SH_z}}
        self.kappaDict = {'fund':self.kappa_f,'SH':self.kappa_SH}
        self.s = np.cross(self.kappaHat,self.z)
        self.p_fm = np.cross(self.s,self.v_fm/self.v_fMag)
        print('Incident kappa: ', self.kappa_f)
        self.p_SHm = np.cross(self.s,self.v_SHm/self.v_SHMag)
        self.p_fp = np.cross(self.s,self.v_fp/self.v_fMag)
        self.p_SHp = np.cross(self.s,self.v_SHp/self.v_SHMag)
        # b, f matrices
        self.b_f = np.column_stack((self.s,self.p_fm))
        self.b_SH = np.column_stack((self.s,self.p_SHm))
        self.f_f = np.column_stack((self.s,self.p_fp))
        print('Incident Field basis (s,p):', self.s, self.p_fm)
        print('Reflected Field basis (s,p):', self.s, self.p_fp)
        self.f_SH = np.column_stack((self.s,self.p_SHp))
        self.bDict = {'fund':self.b_f,'SH':self.b_SH}
        self.fDict = {'fund': self.f_f,'SH':self.f_SH}
        # SH field prefactor
        self.mu = 1.
        self.gamma = (2*np.pi*1j*(self.omega_SH**2)*self.mu)/((constants.c**2)*self.v_SH_z)/np.sqrt(37115548645497.88)

class mediumII():
    def __init__(self,point_group,rotation_matrix,angles_xtal_to_lab):
        self.x = np.array(constants.axes[0])
        self.y = np.array(constants.axes[1])
        self.z = np.array(constants.axes[2])
        self.point_group = point_group
        self.rotation_matrix = rotation_matrix
        self.X2_lab, self.X2_xtal = getX2TensorLab(self.point_group,self.rotation_matrix)
        self.angles_xtal_to_lab = angles_xtal_to_lab
    def assign_eps(self,eps_f,eps_SH):
        self.eps_f = eps_f
        self.eps_SH = eps_SH
        self.epsDict = {'fund':self.eps_f,'SH':self.eps_SH}


class Fresnel:
    def __init__(self,media,fresnel_method,theta,nCmp_jones,phi):
        self.x = np.array(constants.axes[0])
        self.y = np.array(constants.axes[1])
        self.z = np.array(constants.axes[2])
        self.mediumI = media[0]
        self.mediumII = media[1]
        self.mediumIII = media[2]
        self.fresnel_method = fresnel_method
        self.fresnel_dict = {}
        self.theta = theta
        if self.fresnel_method == 'iso_iso':
            self.isotropic_interface = IsotropicInterface()
        elif self.fresnel_method == 'iso_uni':
            self.anisotropic_interface = AnisotropicInterfaceAnalytical(self.mediumII.angles_xtal_to_lab,nCmp_jones,phi)
        elif self.fresnel_method == 'iso_uni_uni':
            self.anisotropic_interface = AnisotropicInterfaceAnalytical(self.mediumII.angles_xtal_to_lab, nCmp_jones,phi)
            self.anisotropic_interface.sol_iso_uni_uni_dict = self.read_iso_uni_uni_soln()


    def read_iso_uni_uni_soln(self):
        # read in transmission/reflection coefficients from text files (solutions output by mathematica code)
        # this function serves to make sure the text is parsed correctly for python
        fresnel_sol_s_in_file = "fresnel_iso_uni_uni_s_in.txt"
        fresnel_sol_p_in_file = "fresnel_iso_uni_uni_p_in.txt"
        files = [fresnel_sol_s_in_file, fresnel_sol_p_in_file]
        symbols = [
            'd', 'K', 'theta', 'q1', 'qo2', 'qe2', 'qo3', 'qe3',
            'Eox2', 'Eoy2', 'Eoz2', 'Eex2', 'Eey2', 'Eez2',
            'Eox3', 'Eoy3', 'Eoz3', 'Eex3', 'Eey3', 'Eez3'
        ]
        symbol_objs = sp.symbols(symbols)
        namespace = {s.name: s for s in symbol_objs}
        namespace.update({'I': sp.I, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos, 'sec': sp.sec})
        sol_dict_s_and_p = {}
        for file in files:
            with open(file, 'r') as f:
                text = f.read()
            if text.startswith("{{") and text.endswith("}}"):
                text = text[2:-2]
                solns = text.split(',')
                coeffs = [string.split(' ->')[0].strip() for string in solns]
                expr_str = [string.split('-> ')[1] for string in solns]
                expr = [sp.sympify(expr_i, locals=namespace) for expr_i in expr_str]
                values_func = [sp.lambdify(symbol_objs, expr_i) for expr_i in expr]
                sol_dict = dict(zip(coeffs, values_func))
                sol_dict_s_and_p.update(sol_dict)
        return sol_dict_s_and_p

    def calc_interface_matrices(self,freq_label,omega):
        # note that the reflection coefficient matrix is denoted 'R' here, but Γ in the publication
        epsI = self.mediumI.epsDict[freq_label]
        v_z_I = self.mediumI.vCmpDict[freq_label]['v_z']
        kappa_mag = self.mediumI.vCmpDict[freq_label]['kappa_mag']
        kappa = self.mediumI.kappaDict[freq_label]
        epsII = self.mediumII.epsDict[freq_label]
        f = self.mediumI.fDict[freq_label]
        b = self.mediumI.bDict[freq_label]
        v0 = calc_v0(omega)
        if self.fresnel_method == 'iso_iso':
            epsII = epsII[0]
            v_z_II = self.isotropic_interface.calc_v_z_iso(eps=epsII, v0=v0, kappa_mag=kappa_mag)
            R,T = self.isotropic_interface.calc_fresnel_mats(epsI,epsII,v_z_I,v_z_II)
            d = self.isotropic_interface.calc_d(self.mediumI.s,kappa,v_z_II,epsII,b,self.z,v0)
            v_z_II = [v_z_II]
            medIII_sym_results = []
        elif self.fresnel_method == 'iso_uni' or self.fresnel_method == 'iso_uni_uni':
            v_z_o = self.anisotropic_interface.calc_v_z_ordinary(epsII,v0,kappa_mag)
            v_z_e = self.anisotropic_interface.calc_v_z_extraordinary(epsII,v0,kappa_mag)
            E_o_unit, E_e_unit = self.anisotropic_interface.calc_uniaxial_fields(epsII,v_z_o,v_z_e,v0,kappa_mag)
            if self.fresnel_method == 'iso_uni_uni':
                epsIII = self.mediumIII.epsDict[freq_label]
                v_z_o_III = self.anisotropic_interface.calc_v_z_ordinary(epsIII, v0, kappa_mag)
                v_z_e_III = self.anisotropic_interface.calc_v_z_extraordinary(epsIII, v0, kappa_mag)
                E_o_unit_III, E_e_unit_III = self.anisotropic_interface.calc_uniaxial_fields(epsIII, v_z_o_III, v_z_e_III,v0, kappa_mag)
                medIII_data = [v_z_o_III,v_z_e_III,E_o_unit_III,E_e_unit_III]
            else:
                medIII_data = []
            R_sym,T_sym,E_o_unit_sym,E_e_unit_sym,v_z_e_sym,medIII_sym_results = self.anisotropic_interface.calc_fresnel_mats(E_o_unit,E_e_unit,kappa_mag,v0,v_z_o,v_z_e,self.theta,epsI,medIII_data,self.mediumII.thickness)
            v_m = self.mediumI.vDict[freq_label]['minus']
            v_p = self.mediumI.vDict[freq_label]['plus']
            if len(medIII_sym_results) == 0:
                R,T,d = self.anisotropic_interface.convert_bases(kappa,v_m,v_p,b,f,constants.axes,R_sym,T_sym,E_o_unit_sym,E_e_unit_sym)
            else:
                R,T,d = R_sym,T_sym,sp.Matrix.hstack(E_o_unit_sym, E_e_unit_sym)
            v_z_II = [v_z_o,v_z_e_sym]
            if len(medIII_sym_results) > 0:
                [R_23_sym, T_23_sym, E_o_unit_III_sym, E_e_unit_III_sym, v_z_e_III_sym] = medIII_sym_results
                R_23,T_23,d_23 = R_23_sym,T_23_sym,sp.Matrix.hstack(E_o_unit_III_sym, E_e_unit_III_sym)
                v_z_III = [v_z_o_III, v_z_e_III_sym]
                self.R_23, self.T_23, self.d_23, self.v_z_III = R_23, T_23, d_23, v_z_III

        self.R = R
        self.T = T
        self.d = d
        self.b = b
        self.f = f
        self.v_z_II = v_z_II
        self.epsII = epsII
        self.kappa = kappa

        wavelength = 1e9 * constants.c / (omega / (2 * np.pi))
        # ensure wavelength list exists and append only if new
        if 'wavelength' not in self.fresnel_dict:
            self.fresnel_dict['wavelength'] = []
        if wavelength not in self.fresnel_dict['wavelength']:
            self.fresnel_dict['wavelength'].append(float(wavelength))

        if freq_label not in self.fresnel_dict:
            self.fresnel_dict[freq_label] = {}

        vars = {'R':self.R,'T':self.T,'d':self.d, 'b':self.b, 'f': self.f, 'v_z_II':self.v_z_II, 'epsII':self.epsII, 'kappa':self.kappa}
        if len(medIII_sym_results) > 0:
            vars23 = {'R_23':self.R_23,'T_23':self.T_23,'d_23':self.d_23,'v_z_III':self.v_z_III}
            vars.update(vars23)
        for key,val in vars.items():
            if key not in self.fresnel_dict[freq_label]:
                self.fresnel_dict[freq_label][key] = []
            self.fresnel_dict[freq_label][key].append(vars[key])


def extract_cos_sin_coeffs(coef_list):
    """
    Convert full complex Fourier coefficients c_{-N..N} to
    isotropic (n=0), cosine (A_n), and sine (B_n) coefficients
    for n >= 1.

    Parameters
    ----------
    coef_list : array_like, shape (2*N+1,)
        Complex Fourier coefficients in order [-N, ..., 0, ..., N]

    Returns
    -------
    A0 : complex
        Isotropic coefficient (n=0)
    A : ndarray, shape (N,)
        Cosine coefficients (complex) for n=1..N
    B : ndarray, shape (N,)
        Sine coefficients (complex) for n=1..N
    """
    N = (len(coef_list) - 1) // 2
    c_neg = coef_list[:N]  # c_{-N..-1}
    c0 = coef_list[N]  # c_0
    c_pos = coef_list[N + 1:]  # c_{1..N}

    # isotropic term
    A0 = c0

    # cosine and sine components
    A = c_pos + c_neg[::-1]  # complex cosine coefficients
    B = 1j * (c_pos - c_neg[::-1])  # complex sine coefficients

    return A0, A, B


def reconstruct_from_cmp(iso,cos_n,sin_n,phi_vals):
    # efficient reconstruction: exp_pos (nPhi, nCmp+1) @ coef_list -> (nPhi,)
    recon = np.full_like(phi_vals, iso, dtype=complex)  # start with isotropic term
    for n, (c, s) in enumerate(zip(cos_n, sin_n), start=1):
        recon += c * np.cos(n * phi_vals) + s * np.sin(n * phi_vals)
    return recon

def reconstruct_from_cmp_sym(iso,cos_n,sin_n,phi):
    recon = iso  # start with isotropic term
    for n, (c, s) in enumerate(zip(cos_n, sin_n), start=1):
        recon += c * sp.cos(n * phi) + s * sp.sin(n * phi)
    return recon

def calc_debye(I=0.1, T=298.15, eps_r=78.5):
    """
    Calculate the Debye length (in meters) for an aqueous electrolyte.

    Parameters
    ----------
    I : float
        Ionic strength of the solution in mol/L.
    T : float, optional
        Temperature in Kelvin. Default is 298.15 K.
    eps_r : float, optional
        Relative permittivity (dielectric constant) of the solvent.
        Default is 78.5 for water at 25°C.

    Returns
    -------
    lambda_D : float
        Debye length in meters.
    """
    # Physical constants
    eps_0 = 8.854e-12       # vacuum permittivity (F/m)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    e = 1.602176634e-19     # elementary charge (C)
    N_A = 6.02214076e23     # Avogadro's number (mol^-1)

    # Convert ionic strength from mol/L to mol/m^3
    I_m3 = I * 1000

    # Bard & Faulkner: λ_D = sqrt( (ε_r ε_0 k_B T) / (2 N_A e^2 I) )
    lambda_D = np.sqrt((eps_r * eps_0 * k_B * T) / (2 * N_A * e**2 * I_m3))
    print('debye length (nm): ',lambda_D*1e9)
    return lambda_D

def load_nk(dir_nk,fname):
    data = np.loadtxt(os.path.join(dir_nk, fname), delimiter=",", skiprows=1)
    return data

def nk_for_model(data,wavelength_selected,plotBool=False):
    wavelength = data[:,0]
    index_arrays_f = []
    index_arrays_SH = []
    for i in range(1,data.shape[1]):
        index_data = data[:,i]
        wavelength_f, index_f, wavelength_SH, index_SH = index_f_and_SH(wavelength,index_data,wavelength_selected,i)
        index_arrays_f.append(index_f)
        index_arrays_SH.append(index_SH)
        if plotBool:
            plt.plot(wavelength,index_data,'o',color='black',label='raw data')
            plt.plot(wavelength_f,index_f,'x',color='red',label='fundamental data')
            plt.plot(wavelength_SH,index_SH,'x',color='blue',label='SH data')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Index')
            plt.title('Optical Constants Read In from File @ Column:' + str(i))
            plt.legend()
            plt.legend()
            plt.show()
    return index_arrays_f, index_arrays_SH


def convert_nk_to_eps(n,k):
    return (n + 1j * k) ** 2

def index_f_and_SH(wavelength,index,wavelength_selected,data_col_idx):
    interp_points = 5000
    wavelength_dense = np.linspace(wavelength[0], wavelength[-1], interp_points)
    interpDense = interp1d(wavelength, index, kind='cubic')

    loc_wavelength_min_SH = np.argmin(wavelength_dense)
    loc_wavelength_max_SH = np.abs(wavelength_dense - np.max(wavelength_dense) / 2).argmin()
    wavelength_SH_maximal = wavelength_dense[loc_wavelength_min_SH : loc_wavelength_max_SH + 1]
    wavelength_f_maximal = 2 * wavelength_SH_maximal
    if data_col_idx == 1:
        print('\nMinimum viable fundamental wavelength: ', np.min(wavelength_f_maximal))
        print('Maximum viable fundamental wavelength: ', np.max(wavelength_f_maximal))
        print('Chosen minimum fundamental wavelength: ', np.min(wavelength_selected))
        print('Chosen maximum fundamental wavelength: ', np.max(wavelength_selected))

    wavelength_f = wavelength_selected
    wavelength_SH = 0.5*wavelength_f
    index_f = interpDense(wavelength_f)
    index_SH = interpDense(wavelength_SH)

    return wavelength_f, index_f, wavelength_SH, index_SH





