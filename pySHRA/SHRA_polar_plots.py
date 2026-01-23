import matplotlib.pyplot as plt
import numpy as np
from SHRAevaluation import field_from_X2_dict, load_result
import os

def plot_grid_polar(fourier_dict,fig_size,wavelength_selected,pol_names=[],X_names=[],fig_name='polar_plot_example.svg',normalization=1):
    print('normalization =',normalization)
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'Arial',
    })
    wavelength = np.array(fourier_dict['wavelength'])
    if len(pol_names) == 0:
        pol_names = [name for name in fourier_dict.keys() if name != 'wavelength']
    if len(X_names) == 0:
        X_names = [name for name in fourier_dict[pol_names[0]]]
    num_X_indices = len(X_names[0].split('_')[1])
    if num_X_indices == 3:
        X_names_pretty = [rf'$F_{{{name.split("_")[1]}}}^\mathrm{{out,in}}$' for name in X_names]
    elif num_X_indices == 4:
        X_names_pretty = [rf'$F_{{{name.split("_")[1]}}}^\mathrm{{out,in}}$' for name in X_names]


    nrows, ncols = len(X_names), len(pol_names)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        subplot_kw={'projection': 'polar'},
        figsize=fig_size,
        squeeze=False
    )

    _ = [ax.set_xticks([]) or ax.set_yticks([]) or ax.grid(False) or ax.spines['polar'].set_visible(False) for ax in axes.flat]
    _ = [axes[0, j].set_title(pol_names[j], pad=10) for j in range(0,len(pol_names))] + \
        [axes[j, 0].set_ylabel(X_names_pretty[j], rotation=0,labelpad=30, va='center') for j in range(nrows)]

    wv_idx = np.argmin(np.abs(wavelength - wavelength_selected))
    phi_vals = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    for i in range(0,len(pol_names)):
        pol_name = pol_names[i]
        for j in range(0,len(X_names)):
            ax = axes[j,i]
            X2_name = X_names[j]
            X2_cmp_dict = fourier_dict[pol_name][X2_name]['components']
            recon = field_from_X2_dict(X2_cmp_dict,wv_idx,phi_vals)/normalization
            intensity = np.real(recon*np.conjugate(recon))
            int_max = np.max(intensity)
            if int_max > 0.0:
                alpha = 1.0
            else:
                alpha = 0.2
            ax.plot(phi_vals,intensity,color='black',alpha=alpha)
            # Convert to scientific notation
            if int_max < 0.01 and int_max > 0:
                exponent = int(np.floor(np.log10(abs(int_max))))
                coeff = int_max / 10 ** exponent
                text_str = rf"${coeff:.1f} \times 10^{{{exponent}}}$"
            else:
                text_str = f"{int_max:.2f}"
            ax.text(0.5,-0.1,text_str,transform=ax.transAxes,ha='center', va='top')
            ax.set_rlim(0.0, 1.1 * int_max)

    fig.subplots_adjust(hspace=1,top=0.83)  # no vertical gap
    plt.show()
    fig.savefig(fig_name)
    return fig, axes

def tabulate_Fijk(fourier_dict,n_list,wavelength_selected,fig_dir,normalization=1):
    print('normalization =', normalization)
    def format_table_entry(mag,phase):
        X_str_n = f"{mag:.2f} ,{round(phase):.0f}°"
        return X_str_n

    wavelength = np.array(fourier_dict['wavelength'])
    wv_idx = np.argmin(np.abs(wavelength - wavelength_selected))
    lines = []
    pol_names = [key for key in fourier_dict.keys() if 'wavelength' not in key]
    X2_names = list(fourier_dict[pol_names[0]].keys())
    for pol_name in pol_names:
        lines.append(pol_name)
        for X2_name in X2_names:
            lines.append(X2_name)
            lines.append(['\t\tn=0','cos(nφ)','\tsin(nφ)'])
            for n_idx in range(0, len(n_list)):
                n = n_list[n_idx]
                row = []
                row.append('n='+str(n))
                X2_cmp_dict = fourier_dict[pol_name][X2_name]['components']
                if n == 0:
                    mag_0 = X2_cmp_dict[n]['iso']['mag'][wv_idx]/normalization
                    phase_0 = X2_cmp_dict[n]['iso']['phase'][wv_idx]*(180/np.pi)
                    iso = format_table_entry(mag_0, phase_0)
                    cos = ''
                    sin = ''
                if n > 0:
                    mag_cn = X2_cmp_dict[n]['cos']['mag'][wv_idx]/normalization
                    phase_cn = X2_cmp_dict[n]['cos']['phase'][wv_idx] * (180 / np.pi)
                    mag_sn = X2_cmp_dict[n]['sin']['mag'][wv_idx]/normalization
                    phase_sn = X2_cmp_dict[n]['sin']['phase'][wv_idx] * (180 / np.pi)
                    cos = format_table_entry(mag_cn, phase_cn)
                    sin = format_table_entry(mag_sn, phase_sn)
                    iso = ''
                row.append(iso)
                row.append(cos)
                row.append(sin)
                lines.append(row)
        lines.append('*' * 100)

    def write_list_to_file(data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                if isinstance(item, list):
                    f.write("\t\t".join(str(x) for x in item) + "\n")
                else:
                    f.write(str(item) + "\n")

    write_list_to_file(lines, fig_dir + "Fijk_tabulated.txt")




output_dir = "C:/Users/...CHANGETHIS..." # this should be the folder where the output files live
fig_dir = output_dir # plot_grid_polar will save a svg to this folder
output_files = [f for f in os.listdir(output_dir) if 'fourier' in f]
print(output_files)
file_selected = output_files[0]
print(file_selected)
X2_fourier_dict = load_result(output_dir + file_selected)
fig_name = os.path.join(fig_dir, 'polar_plot_example.svg')
normalization = np.sqrt(37115548645497.88)  # normalize to max(F_333) for RuO2(110)-argon interface

# plot_grid_polar takes the square modulus of each F_ijk(φ) and displays it on a polar plot
# The subplots for all X_ijk and input-output combination are combined into one figure
plot_grid_polar(X2_fourier_dict,fig_size=[4,3],wavelength_selected=800,fig_name=fig_name,normalization=normalization)
# tabulate_Fijk writes a text file containing the F_ijk (magnitude and phase) for each X_ijk and input-output combination
# The table is broken down by Fourier component, which is labeled with 'n'
tabulate_Fijk(X2_fourier_dict,n_list=[0,2,4],wavelength_selected=800,fig_dir=fig_dir,normalization=normalization)


