import numpy as np
from SHRAhelper import mediumI, mediumII, Fresnel, Constants, RotationMatrices
from SHRAhelper import load_nk, nk_for_model, convert_nk_to_eps, index_f_and_SH
from SHRAevaluation import Evaluation, save_result
from refractiveindex import RefractiveIndexMaterial
import os

constants = Constants()
rotation_mats = RotationMatrices()


def run_model(output_dir, point_group_I, point_group_II, sources, optical_data_II, info_label, optical_data_III=[],medII_thickness=np.inf,fresnel_method='iso_uni'):
    [eps_I_f, eps_I_SH, eps_o_f, eps_e_f, eps_o_SH, eps_e_SH, wavelengths] = optical_data_II
    if len(optical_data_III)> 0:
        [eps_o_f_III, eps_e_f_III, eps_o_SH_III, eps_e_SH_III] = optical_data_III
    # **********************************************************************
    # Setup
    theta_deg = 45
    theta = theta_deg*np.pi/180
    print('Angle of incidence (deg): ', theta_deg)
    input('Treating system as: ' + fresnel_method + '. Press enter to continue.')
    Q_z = rotation_mats.Q_z # rotation matrix for azimuthal rotation about z axis
    phi = rotation_mats.phi
    pi = rotation_mats.pi
    rotation_matrixI = Q_z
    rotation_matrixII = Q_z
    nCmp_jones = 8 # number of components in Fourier decomposition for transmission/reflection coefficients
    angles_xtal_to_lab = [phi,(pi/2)-phi,pi/2] # angles between optic axis 'c' and lab axes (x,y,z)
    medI = mediumI(theta,point_group=point_group_I,rotation_matrix=rotation_matrixI)
    medII = mediumII(point_group=point_group_II,rotation_matrix=rotation_matrixII,angles_xtal_to_lab=angles_xtal_to_lab)
    if len(optical_data_III) > 0 and fresnel_method == 'iso_uni_uni':
        medII.thickness = medII_thickness
        medIII = mediumII(point_group=point_group_II,rotation_matrix=rotation_matrixII,angles_xtal_to_lab=angles_xtal_to_lab)
    else:
        medII.thickness = np.inf
        medIII = []
    media = [medI, medII, medIII]
    fresnel_f = Fresnel(media=media,fresnel_method=fresnel_method,theta=theta,nCmp_jones=nCmp_jones,phi=phi)
    fresnel_SH = Fresnel(media=media,fresnel_method=fresnel_method,theta=theta,nCmp_jones=nCmp_jones,phi=phi)
    a_dict = {'s-in': np.array([1, 0]), 'p-in': np.array([0, 1])} # defines input Jones vector
    q_hat_dict = {'s-out': np.array([]),'p-out': np.array([])} # defines output Jones vector
    nCmp = 10 # number of components in Fourier decomposition for SH field
    evaluation = Evaluation(media,fresnel_f,fresnel_SH,a_dict,q_hat_dict,nCmp)

    # **********************************************************************
    # Evaluate SH fields at chosen wavelengths
    for i in range(0,len(wavelengths)):
        print('Current wavelength: ', wavelengths[i])
        wavelength = wavelengths[i]*1e-9
        omega_f = 2*np.pi*constants.c/wavelength
        # optical constants of isotropic medium
        eps_I_f_i = eps_I_f[i]
        eps_I_SH_i = eps_I_SH[i]
        # optical constants of anisotropic medium
        eps_o_f_i = eps_o_f[i]
        eps_e_f_i = eps_e_f[i]
        eps_o_SH_i = eps_o_SH[i]
        eps_e_SH_i = eps_e_SH[i]
        eps_II_f_i = [eps_o_f_i,eps_e_f_i]
        eps_II_SH_i = [eps_o_SH_i,eps_e_SH_i]
        # optical constants of substrate
        if len(optical_data_III) > 0 and fresnel_method == 'iso_uni_uni':
            eps_o_f_III_i = eps_o_f_III[i]
            eps_e_f_III_i = eps_e_f_III[i]
            eps_o_SH_III_i = eps_o_SH_III[i]
            eps_e_SH_III_i = eps_e_SH_III[i]
            eps_III_f_i = [eps_o_f_III_i, eps_e_f_III_i]
            eps_III_SH_i = [eps_o_SH_III_i, eps_e_SH_III_i]
            evaluation.medIII.assign_eps(eps_f=eps_III_f_i, eps_SH=eps_III_SH_i)

        evaluation.medI.setup(omega_f=omega_f,eps_f=eps_I_f_i,eps_SH=eps_I_SH_i)
        evaluation.medII.assign_eps(eps_f=eps_II_f_i,eps_SH=eps_II_SH_i)

        evaluation.evaluateSH(sources)

    # **********************************************************************
    # Save results
    info_label = info_label + '_'
    fund_fresnel_filename = output_dir + info_label + 'fund_fresnel'
    SH_fresnel_filename = output_dir + info_label + 'SH_fresnel'
    X2_fourier_filename = output_dir + info_label + 'X2_fourier'
    X3_fourier_filename = output_dir + info_label + 'X3_fourier'
    save_result(evaluation.X2_result,X2_fourier_filename)
    save_result(evaluation.X3_result,X3_fourier_filename)
    save_result(fresnel_f.fresnel_dict,fund_fresnel_filename)
    save_result(fresnel_SH.fresnel_dict, SH_fresnel_filename)
    print('Success. Output files have been generated.')


def setup_optical_const(dir_nk,read_file_for_II,read_file_for_I,read_file_for_III,medium_II_name='',medium_I_name='',medium_III_name='',wv_bounds=[]):
    # **********************************************************************
    # Read in data from refractiveindex.info
    wavelength_db = np.linspace(400,1000,5000) # initial wl array (in nm) just for accessing database
    Pt = RefractiveIndexMaterial(shelf='main', book='Pt', page='Tselin')
    silica = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Malitson')
    air = RefractiveIndexMaterial(shelf='other', book='air', page='Ciddor')
    water = RefractiveIndexMaterial(shelf='main', book='H2O', page='Daimon-21.5C')
    argon = RefractiveIndexMaterial(shelf='main', book='Ar', page='Borzsonyi')
    n_Pt = Pt.get_refractive_index(wavelength_db)
    k_Pt = Pt.get_extinction_coefficient(wavelength_db)
    n_argon = argon.get_refractive_index(wavelength_db)
    k_argon = np.zeros(len(n_argon))
    n_air = air.get_refractive_index(wavelength_db)
    k_air = np.zeros(len(n_air))
    n_water = water.get_refractive_index(wavelength_db)
    k_water = np.zeros(len(n_water))
    n_silica = silica.get_refractive_index(wavelength_db)
    k_silica = np.zeros(len(n_silica))
    index_dict = {'Pt': {'n':n_Pt,'k':k_Pt},'air':{'n':n_air,'k':k_air},'argon':{'n':n_argon,'k':k_argon},'water':{'n':n_water,'k':k_water},'silica':{'n':n_silica,'k':k_silica}}
    # **********************************************************************
    # Optionally, set file name for medium II
    if read_file_for_II:
        nk_files = [f for f in os.listdir(dir_nk) if medium_II_name in f]
        print('\nnk_files matching medium_II_name: ', nk_files)
        nk_file_II = nk_files[0] # replace with correct file index
        print('\nnk_file_II: ', nk_file_II)
        medium_II_name = nk_file_II.split('_')[0]
    else:
        print('\nRetrieving medium II (isotropic) optical constants from database...')
        n_II = index_dict[medium_II_name]['n']
        k_II = index_dict[medium_II_name]['k']
    # **********************************************************************
    # Optionally, set file name for medium III
    if read_file_for_III and medium_III_name != '':
        nk_files = [f for f in os.listdir(dir_nk) if medium_III_name in f]
        print('\nnk_files matching medium_III_name: ', nk_files)
        nk_file_III = nk_files[0] # replace with correct file index
        print('\nnk_file_III: ', nk_file_III)
        medium_III_name = nk_file_III.split('_')[0]
    elif read_file_for_III == False and medium_III_name != '':
        print('\nRetrieving medium III (isotropic) optical constants from database...')
        n_III = index_dict[medium_III_name]['n']
        k_III = index_dict[medium_III_name]['k']
    # **********************************************************************
    # Optionally, set file name for medium I
    if read_file_for_I == True:
        nk_files = [f for f in os.listdir(dir_nk) if medium_I_name in f]
        print('\nnk_files matching medium_I_name: ', nk_files)
        nk_file_I = nk_files[0] # replace with correct file index
        print('\nnk_file_I: ', nk_file_I)
        medium_I_name = nk_file_I.split('_')[0]
    else:
        print('\nRetrieving medium I optical constants from database...')
        n_I = index_dict[medium_I_name]['n']
        k_I = index_dict[medium_I_name]['k']
    # **********************************************************************
    # Select wavelengths at which to run model.
    if len(wv_bounds)>0:
        wv_bounds = wv_bounds
    else:
        wv_bounds = [800,800]
    wavelength_start = wv_bounds[0]
    wavelength_end = wv_bounds[1]
    step = 10
    wavelength_selected = np.arange(wavelength_start,wavelength_end+step,step)
    print('wavelengths for model: ', wavelength_selected)
    # **********************************************************************
    # Read in data for medium I and construct optical const arrays @ fundamental and SH
    if read_file_for_I == True:
        nk_data_I = load_nk(dir_nk,nk_file_I)
    else:
        nk_data_I = np.column_stack((wavelength_db, n_I, k_I))

    plotBool = False # set to true to verify interpolation of optical constants is done correctly
    index_arrays_I_f, index_arrays_I_SH = nk_for_model(nk_data_I, wavelength_selected, plotBool=plotBool)
    n_I_f, k_I_f = index_arrays_I_f[0], index_arrays_I_f[1]
    n_I_SH, k_I_SH = index_arrays_I_SH[0], index_arrays_I_SH[1]
    eps_I_f = convert_nk_to_eps(n_I_f,k_I_f)
    eps_I_SH = convert_nk_to_eps(n_I_SH,k_I_SH)
    # **********************************************************************
    test_factor = 1#(1+1e-4)
    # Sanity check: in this model, all fresnel methods ("iso_iso", "iso_uni", "iso_ani_ani") should produce the same
    # observables (specifically, the same *total* transmitted and reflected fields) in the limit of isotropic materials.
    # However, the "iso_uni_uni" method relies on long unsimplified expressions and will produce 'nan' when simulating
    # iso-iso-iso interfaces unless you introduce some small birefringence with a test_factor != 1.

    # Read in data for medium II and construct optical const arrays @ fundamental and SH
    if read_file_for_II == True:
        nk_data_II = load_nk(dir_nk,nk_file_II)
        index_arrays_II_f, index_arrays_II_SH = nk_for_model(nk_data_II,wavelength_selected,plotBool=plotBool)
        n_o_f, k_o_f, n_e_f, k_e_f = index_arrays_II_f[0], index_arrays_II_f[1], index_arrays_II_f[2], index_arrays_II_f[3]
        n_o_SH, k_o_SH, n_e_SH, k_e_SH = index_arrays_II_SH[0], index_arrays_II_SH[1], index_arrays_II_SH[2], index_arrays_II_SH[3]
    else:
        nk_data_II = np.column_stack((wavelength_db, n_II, k_II))
        index_arrays_II_f, index_arrays_II_SH = nk_for_model(nk_data_II, wavelength_selected, plotBool=plotBool)
        n_o_f, k_o_f = index_arrays_II_f[0], index_arrays_II_f[1]
        n_o_SH, k_o_SH = index_arrays_II_SH[0], index_arrays_II_SH[1]
        n_e_f = n_o_f*(test_factor)# set extraordinary equal to ordinary (assumes n,k pulled from database are isotropic)
        k_e_f = k_o_f*(test_factor)
        n_e_SH = n_o_SH*(test_factor)
        k_e_SH = k_o_SH*(test_factor)

    eps_o_f = convert_nk_to_eps(n_o_f,k_o_f)
    eps_e_f = convert_nk_to_eps(n_e_f,k_e_f)
    eps_o_SH = convert_nk_to_eps(n_o_SH,k_o_SH)
    eps_e_SH = convert_nk_to_eps(n_e_SH,k_e_SH)

    # **********************************************************************
    # Read in data for medium III and construct optical const arrays @ fundamental and SH
    if read_file_for_III == True and medium_III_name !='':
        nk_data_III = load_nk(dir_nk, nk_file_III)
        index_arrays_III_f, index_arrays_III_SH = nk_for_model(nk_data_III, wavelength_selected, plotBool=plotBool)
        n_o_f_III, k_o_f_III, n_e_f_III, k_e_f_III = index_arrays_III_f[0], index_arrays_III_f[1], index_arrays_III_f[2], \
        index_arrays_III_f[3]
        n_o_SH_III, k_o_SH_III, n_e_SH_III, k_e_SH_III = index_arrays_III_SH[0], index_arrays_III_SH[1], index_arrays_III_SH[2], \
        index_arrays_III_SH[3]
    elif read_file_for_III == False and medium_III_name !='':
        nk_data_III = np.column_stack((wavelength_db, n_III, k_III))
        index_arrays_III_f, index_arrays_III_SH = nk_for_model(nk_data_III, wavelength_selected, plotBool=plotBool)
        n_o_f_III, k_o_f_III = index_arrays_III_f[0], index_arrays_III_f[1]
        n_o_SH_III, k_o_SH_III = index_arrays_III_SH[0], index_arrays_III_SH[1]
        n_e_f_III = n_o_f_III*(test_factor) # set extraordinary equal to ordinary (assumes n,k pulled from database are isotropic)
        k_e_f_III = k_o_f_III*(test_factor)
        n_e_SH_III = n_o_SH_III*(test_factor)
        k_e_SH_III = k_o_SH_III*(test_factor)

    if medium_III_name != "":
        eps_o_f_III = convert_nk_to_eps(n_o_f_III, k_o_f_III)
        eps_e_f_III = convert_nk_to_eps(n_e_f_III, k_e_f_III)
        eps_o_SH_III = convert_nk_to_eps(n_o_SH_III, k_o_SH_III)
        eps_e_SH_III = convert_nk_to_eps(n_e_SH_III, k_e_SH_III)
        optical_data_III = [eps_o_f_III, eps_e_f_III, eps_o_SH_III, eps_e_SH_III]
    else:
        optical_data_III = []
    # **********************************************************************

    optical_data = [eps_I_f, eps_I_SH, eps_o_f, eps_e_f, eps_o_SH, eps_e_SH, wavelength_selected]
    return optical_data, medium_II_name, medium_I_name, medium_III_name, optical_data_III
# **********************************************************************
dir_nk = "C:/Users/...CHANGETHIS..." # change to directory where n,k files are located
output_dir = "C:/Users/...CHANGETHIS..." # change to directory where output files should go

measured_mats = ['TiO2','RuO2','IrO2'] # read in n,k for these materials

# assign point groups of materials
point_group_dict = {'argon': 'infm','air': 'infm', 'water': 'infm', 'RuO2': 'mm2', 'IrO2': 'mm2',
                    'TiO2': 'mm2', 'Pt': 'infm','silica': 'infm'}

# select wavelength bounds for simulation (in nm)
wv_bounds_dict = {'RuO2': [800,800], 'IrO2': [800,800], 'TiO2': [800,800], 'Pt': [800,800],'silica': [800,800]}

media_II = ['RuO2']
medium_I_name = 'water'
medium_III_name = 'TiO2'
medII_thickness = 9 # thickness in nm
for j in range(0,len(media_II)):
    print('\nRunning model for medium II: ', media_II[j])
    print('\nMedium I: ', medium_I_name)
    medium_II_name = media_II[j]
    if medium_I_name not in measured_mats:
        read_file_for_I = False
    else:
        read_file_for_I = True
    if medium_II_name not in measured_mats:
        read_file_for_II = False
    else:
        read_file_for_II = True
    if medium_III_name not in measured_mats:
        read_file_for_III = False
    else:
        read_file_for_III = True
    optical_data, medium_II_name, medium_I_name, medium_III_name, optical_data_III = setup_optical_const(
        dir_nk=dir_nk,
        read_file_for_II=read_file_for_II,
        read_file_for_I=read_file_for_I,
        read_file_for_III=read_file_for_III,
        medium_II_name=medium_II_name,
        medium_I_name=medium_I_name,
        medium_III_name=medium_III_name,
        wv_bounds=wv_bounds_dict[medium_II_name])

    # Run the model
    point_group_I = point_group_dict[medium_I_name]
    point_group_II = point_group_dict[medium_II_name]
    info_label = medium_II_name + '_' + point_group_II + '_' + str(medII_thickness) + 'nm' + '_' + medium_I_name + '_' + point_group_I
    info_label = info_label + '_' + medium_III_name
    print('\nPrefix for output files: ', info_label)
    sources = ['surface','EFISH']
    fresnel_methods = ['iso_iso', 'iso_uni', 'iso_uni_uni']
    fresnel_method = fresnel_methods[2]
    run_model(output_dir, point_group_I, point_group_II, sources, optical_data, info_label, optical_data_III, medII_thickness, fresnel_method)










