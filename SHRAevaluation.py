import numpy as np
import sympy as sp
from sympy import lambdify
from SHRAhelper import Constants, extract_cos_sin_coeffs, reconstruct_from_cmp, calc_debye
import pickle
import matplotlib.pyplot as plt



def save_result(result_dict, filename):
    """Save result dictionary to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def prune_small_terms_matrix(M, tol=1e-8):
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


def load_result(filename):
    """Load result dictionary from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Evaluation:
    def __init__(self,media,fresnel_f,fresnel_SH, a_dict, q_hat_dict,nCmp):
        self.medI = media[0]
        self.medII = media[1]
        self.medIII = media[2]
        self.fresnel_f = fresnel_f
        self.fresnel_SH = fresnel_SH
        self.a_dict = a_dict
        self.q_hat_dict = q_hat_dict
        self.X2_list = sorted({str(chi) for chi in self.medII.X2_xtal.free_symbols})
        self.X3_list = sorted({str(chi) for chi in self.medI.X3_xtal.free_symbols})
        self.nCmp = nCmp
        self.precompute_fourier_matrix(nCmp=self.nCmp)
        self.X2_result = {}
        self.X3_result = {}
        self.constants = Constants()

    def evaluateSH(self,sources):
        omega_f, omega_SH = self.medI.omega_f, self.medI.omega_SH
        self.fresnel_f.calc_interface_matrices('fund', omega_f)
        self.fresnel_SH.calc_interface_matrices('SH', omega_SH)
        R_f, T_f, d_f = self.fresnel_f.R, self.fresnel_f.T, self.fresnel_f.d
        R_SH, T_SH, d_SH = self.fresnel_SH.R, self.fresnel_SH.T, self.fresnel_SH.d
        gamma, b_f, b_SH, f_f, f_SH = self.medI.gamma, self.medI.b_f, self.medI.b_SH, self.medI.f_f, self.medI.f_SH
        B = f_SH @ np.transpose(f_SH)
        C = f_SH @ R_SH @ np.transpose(b_SH)
        X2_lab = self.medII.X2_lab
        s, p_SHp = self.medI.s, self.medI.p_SHp
        v_z_I_f = self.medI.vCmpDict['fund']['v_z']
        v_z_I_SH = self.medI.vCmpDict['SH']['v_z']
        X3_lab = self.medI.X3_lab
        for key,val in self.q_hat_dict.items():
            if key == 's-out':
                self.q_hat_dict['s-out'] = s
            elif key == 'p-out':
                self.q_hat_dict['p-out'] = p_SHp
        phi = sp.Symbol('phi',real=True)
        trans_s_in = sp.Matrix(d_f @ T_f @ np.array([1,0]))
        trans_p_in = sp.Matrix(d_f @ T_f @ np.array([0,1]))
        ref_s_in = sp.Matrix(f_f @ R_f @ np.array([1,0]))
        ref_p_in = sp.Matrix(f_f @ R_f @ np.array([0,1]))
        trans_s_in = [x.evalf(subs={phi: np.pi/3}) for x in trans_s_in]
        trans_p_in = [x.evalf(subs={phi: np.pi/3}) for x in trans_p_in]
        ref_s_in = [x.evalf(subs={phi: np.pi / 3}) for x in ref_s_in]
        ref_p_in = [x.evalf(subs={phi: np.pi / 3}) for x in ref_p_in]
        print('transmitted fundamental with s input:\n', trans_s_in)
        print('transmitted fundamental with p input:\n', trans_p_in)
        print('reflected fundamental with s input:\n', ref_s_in)
        print('reflected fundamental with p input:\n', ref_p_in)
        if 'surface' in sources:
            for q_hat_key, q_hat_val in self.q_hat_dict.items():
                for a_key, a_val in self.a_dict.items():
                    E_q_II = gamma * np.einsum('h,hj,jrs,rm,mn,n,so,ol,l',
                                               q_hat_val, B + C, X2_lab, d_f, T_f, a_val, d_f, T_f, a_val)

                    field_label = a_key + '/' + q_hat_key
                    self.get_fourier_cmps(field=E_q_II,X_list=self.X2_list,nCmp=self.nCmp,omega=self.medI.omega_f,result=self.X2_result,field_label=field_label,source='surface')

        if 'EFISH' in sources:
            X3_cont = X3_lab[:, :, :, 2]
            deb = calc_debye()
            surface_potential = 1
            F1 = 1 / (1 + 1j * deb * (2*v_z_I_f + v_z_I_SH))
            F2 = 1 / (1 - 1j * deb * (2*v_z_I_f - v_z_I_SH))
            F3 = 1 / (1 + 1j * deb * (v_z_I_SH))
            F4 = 1 / (1 - 1j * deb * (-2 * v_z_I_f + v_z_I_SH))
            F5 = 1 / (1 - 1j * deb * (2 * v_z_I_f + v_z_I_SH))
            F6 = 1 / (1 - 1j * deb * (v_z_I_SH))
            for q_hat_key, q_hat_val in self.q_hat_dict.items():
                for a_key, a_val in self.a_dict.items():
                    b_a = np.einsum('kn,n->k', b_f, a_val)  # b_kn a_n
                    fRa = np.einsum('km,mn,n->k', f_f, R_f, a_val)  # f_km Γ_mn a_n

                    # --- S matrices, all (k,l) ---
                    S1 = np.einsum('k,l->kl', b_a, b_a)  # b_kn a_n * b_lp a_p
                    S2 = np.einsum('k,l->kl', fRa, fRa)  # f_kmΓ_mn a_n * f_lrΓ_rp a_p
                    S3 = np.einsum('k,l->kl', b_a, fRa) + np.einsum('k,l->kl', fRa, b_a)
                    # now: S3[k,l] = b_kn a_n f_lrΓ_rp a_p + f_kmΓ_mn a_n b_lp a_p

                    # --- χ contraction over k,l -> (j,)
                    T1 = np.einsum('jkl,kl->j', X3_cont, S1)
                    T2 = np.einsum('jkl,kl->j', X3_cont, S2)
                    T3 = np.einsum('jkl,kl->j', X3_cont, S3)

                    # --- B,C contractions over j -> (h,)
                    E_B_h = np.einsum('hj,j->h', B, F1 * T1 + F2 * T2 + F3 * T3)
                    E_C_h = np.einsum('hj,j->h', C, F4 * T1 + F5 * T2 + F6 * T3)

                    # --- projection with q̂_h and prefactors -> scalar per q
                    E_q_EFISH_total = gamma * surface_potential * np.einsum('h,h->', q_hat_val, E_B_h + E_C_h)
                    field_label = a_key + '/' + q_hat_key
                    self.get_fourier_cmps(field=E_q_EFISH_total, X_list=self.X3_list, nCmp=self.nCmp, omega=self.medI.omega_f,
                                          result=self.X3_result, field_label=field_label, source='EFISH')

    def precompute_fourier_matrix(self, nCmp, nPhi=1000):
        self.phi_vals = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
        n_array = np.arange(-nCmp, nCmp + 1).reshape(-1, 1)  # includes negative freqs
        self.fourier_matrix = np.exp(-1j * n_array * self.phi_vals)  # (2nCmp+1, nPhi)

    def get_fourier_cmps(self, field, X_list, nCmp, omega, result, field_label, source):
        """
        Compute Fourier components for each chi in X_list at frequency omega,
        using precomputed self.phi_vals and self.fourier_matrix.

        - field: sympy.Expr (may be sympy.Zero) that depends on phi and chi symbols
        - X_list: list of chi identifiers (strings or sympy.Symbols)
        - nCmp: max harmonic index (0..nCmp)
        - omega: angular frequency (rad/s)
        - result: dict to append results into (persistent)
        - field_label: str
        """

        # wavelength (nm)
        wavelength = 1e9 * self.constants.c / (omega / (2 * np.pi))

        # ensure wavelength list exists and append only if new
        if 'wavelength' not in result:
            result['wavelength'] = []
        if wavelength not in result['wavelength']:
            result['wavelength'].append(float(wavelength))

        # ensure field_label exists
        if field_label not in result:
            result[field_label] = {}

        # phi symbol for lambdify/subs
        phi = sp.Symbol('phi')

        # precomputed arrays
        phi_vals = self.phi_vals  # shape (nPhi,)
        fourier_matrix = self.fourier_matrix  # shape (nCmp+1, nPhi)
        nPhi = len(phi_vals)

        # create sympy symbol list for substitution (use names)
        sym_list = [sp.Symbol(str(x)) for x in X_list]
        fast_func = lambdify([phi, *sym_list], field, 'numpy')

        # iterate chi elements
        for idx, chi in enumerate(X_list):
            chi_key = str(chi)  # key stored in dict

            # init structure for this chi if missing
            if chi_key not in result[field_label]:
                result[field_label][chi_key] = {
                    'RRMSE': [],
                    'components': {
                        0: {
                            'iso': {'mag': [], 'phase': []}},
                            **{n: {'cos': {'mag': [], 'phase': []},
                                   'sin': {'mag': [], 'phase': []}}
                               for n in range(1, nCmp + 1)}
                        }
                    }

            X_vals = np.zeros(len(sym_list))
            X_vals[idx] = 1

            # Evaluate field (vectorized) for this chi substitution
            if field == 0 or field == sp.S.Zero:
                field_vals = np.zeros(nPhi, dtype=complex)
            else:
                field_vals = fast_func(phi_vals, *X_vals)

                if np.isscalar(field_vals):
                    field_vals = np.full_like(phi_vals, field_vals, dtype=complex)
                else:
                    field_vals = np.array(field_vals, dtype=complex)

            # If field is all zeros, shortcut the Fourier coefficients
            if np.allclose(field_vals, 0,atol=(1+1j)*1e-30):
                coef_list = np.zeros(2*nCmp + 1, dtype=complex)
            else:
                print(chi_key,field_label,np.max(field_vals))
                # if chi_key =='chi_113' and field_label == 'p-in/s-out':
                #     import matplotlib.pyplot as plt
                #     plt.plot(np.real(field_vals))
                #     plt.plot(np.imag(field_vals))
                #     plt.show()
                # if chi_key =='chi_311' and field_label == 'p-in/p-out':
                #     import matplotlib.pyplot as plt
                #     plt.plot(np.real(field_vals))
                #     plt.plot(np.imag(field_vals))
                #     plt.show()
                # # matrix (nCmp+1, nPhi) @ vector (nPhi,) -> (nCmp+1,)
                coef_list = (fourier_matrix @ field_vals) / float(nPhi)

            iso, cos_n, sin_n = extract_cos_sin_coeffs(coef_list)
            # magnitudes and phases appended to lists (so they grow over wavelengths)
            mag = float(np.abs(iso))
            ph = float(np.angle(iso))
            result[field_label][chi_key]['components'][0]['iso']['mag'].append(mag)
            result[field_label][chi_key]['components'][0]['iso']['phase'].append(ph)
            for n in range(0,len(cos_n)):
                mag = float(np.abs(cos_n[n]))
                ph = float(np.angle(cos_n[n]))
                result[field_label][chi_key]['components'][n+1]['cos']['mag'].append(mag)
                result[field_label][chi_key]['components'][n+1]['cos']['phase'].append(ph)
            for n in range(0,len(sin_n)):
                mag = float(np.abs(sin_n[n]))
                ph = float(np.angle(sin_n[n]))
                result[field_label][chi_key]['components'][n+1]['sin']['mag'].append(mag)
                result[field_label][chi_key]['components'][n+1]['sin']['phase'].append(ph)

            # reconstruct and compute RRMSE
            if np.allclose(field_vals, 0):
                rrmse = 0.0
            else:
                recon = reconstruct_from_cmp(iso,cos_n,sin_n,phi_vals)
                num = np.mean(np.abs(field_vals - recon) ** 2)
                den = np.mean(np.abs(field_vals) ** 2)
                rrmse = float(np.sqrt(num) / (np.sqrt(den) + 1e-30))
                if np.abs(rrmse) > 0.01:
                #if chi_key == 'chi_333':
                #if True:
                    print('Increase number of fourier components to improve accuracy...')
                    plt.plot(self.phi_vals, np.abs(field_vals), color='black')
                    plt.plot(self.phi_vals, np.abs(recon), color='red')
                    plt.title('Reconstruction for: ' + field_label + ', ' + chi_key)
                    plt.show()

            result[field_label][chi_key]['RRMSE'].append(rrmse)

        # persist back to self for batch runs
        if source == 'surface':
            self.X2_result = result
        elif source == 'EFISH':
            self.X3_result = result

        return result


def field_from_X2_dict(X2_cmp_dict,wv_idx,phi_vals):
    nCmp = len(X2_cmp_dict)
    iso = (X2_cmp_dict[0]['iso']['mag'][wv_idx]) * np.exp(1j * X2_cmp_dict[0]['iso']['phase'][wv_idx])
    cos_terms = []
    sin_terms = []
    for n in range(1, nCmp):
        cos_terms.append((X2_cmp_dict[n]['cos']['mag'][wv_idx]) * np.exp(1j * X2_cmp_dict[n]['cos']['phase'][wv_idx]))
        sin_terms.append((X2_cmp_dict[n]['sin']['mag'][wv_idx]) * np.exp(1j * X2_cmp_dict[n]['sin']['phase'][wv_idx]))
    recon = reconstruct_from_cmp(iso, cos_terms, sin_terms, phi_vals)
    return recon
