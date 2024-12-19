import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import stats as st
from sklearn.metrics.cluster import normalized_mutual_info_score as calcMI

def main(file, reffile, outpath, col_or_lin, clas, test = "", invert = True):
    """
    Executes a full iteration of the workflow, including data processing,
    SVD, and mutual information calculations.

    Parameters
    ----------
    file : str
        File path to the spectral data file.
    reffile : str
        File path to the file mapping spectra to their respective classes.
    col_or_lin : str
        Specifies the organization of the data: "colonnes", "lignes", or "CDToolX".
    clas : str
        The class to iterate on.
    test : str, optional
        An additional class for comparison. Default is an empty string.
    invert : bool, optional
        If True, corrects an inverted spectrum. Default is False.

    Returns
    -------
    None
    """
    wl, data, label, target = open_data(file, reffile, col_or_lin)
    r_data, r_target, r_label = class_data(wl, data, label, clas, target)
    for i in range(len(r_target)):
        print(r_label[i], r_target[i])

    basis, df_c, df_p = SVD_on_class(r_data, clas, data, label, wl[0], invert)
    r_data_, r_target_, r_label = class_data(wl, data, label, "_", target)
    MIs = MI_on_class("_", basis, data, label, wl[0])
    write_csv(f"{outpath}/{test}_vs_{clas}", MIs, df_c["Basis # 0"], label)


def open_data(file, reffile, col_or_lin):
    """
    Loads and organizes spectral data from the provided files.

    Parameters
    ----------
    file : str
        File path to the spectral data file.
    reffile : str
        File path to the reference file mapping spectra to classes.
    col_or_lin : str
        Specifies the organization of the data: "colonnes", "lignes", or "CDToolX".

    Returns
    -------
    n_wl : list of list of float
        List of wavelength lists for each spectrum.
    n_data : list of list of float
        List of intensity values for each spectrum.
    label : list of str
        List of spectrum names.
    target : dict
        Dictionary mapping spectrum names to their classes.
    """
    if col_or_lin == "colonnes":
        with open(file, encoding = "windows-1252") as filin: #open text file
            line = [] #read line by line and split line sep by ";"
            for i in filin.readlines():
                line.append(i.split(";"))

        target = {}
        with open(reffile, "r", encoding = "windows-1252") as filin:
            lignes = filin.readlines()
            for i in lignes[1:]:
                target[i.split(";")[0]] = i.split(";")[1].strip()
        print(target)
                
        label = [] #read each line and add new label in label
        for i in line[0][1:]:
            if i != '\n':
                if i not in label and i in list(target.keys()):
                    label.append(i)
        print(label)


        data = [] #for each label find wavelenght and values
        wl = []
        for j in range(len(line[0][1:])):
            tmp_wl = []
            tmp_data = []
            if line[0][j+1] in label:
                for k in line[1:]:
                    tmp_wl.append(float(k[0].strip()))
                    tmp_data.append(float(k[j+1].strip()))
                wl.append(tmp_wl)
                data.append(tmp_data)

    elif col_or_lin == "lignes":
        with open(file, encoding = "windows-1252") as filin: #open text file
            line = [] #read line by line and split line sep by ";"
            for i in filin.readlines():
                line.append(i.split(";"))

        target = {}
        with open(reffile, "r", encoding = "windows-1252") as filin:
            lignes = filin.readlines()
            for i in lignes[1:]:
                target[i.split(";")[0]] = i.split(";")[1].strip()
        print(target.keys())
                
        label = [] #read each line and add new label in label
        for i in line[1:]:
            if i[0] != '\n':
                if i[0] not in label and i[0] in list(target.keys()):
                    label.append(i[0])

        data = [] #for each label find wavelenght and values
        wl = []
        for j in line[1:]:
            tmp_wl = []
            tmp_data = []
            if j[0] in label:
                for i in range(len(j[1:])):
                    tmp_wl.append(float(line[0][i+1].strip()))
                    tmp_data.append(float(j[i+1].strip()))
                wl.append(tmp_wl)
                data.append(tmp_data)


    elif col_or_lin == "CDToolX":
        with open(file, encoding = "windows-1252") as filin: #open text file
            line = [] #read line by line and split line sep by ";"
            for i in filin.readlines():
                line.append(i.split(";"))

        target = {}
        with open(reffile, "r", encoding = "windows-1252") as filin:
            lignes = filin.readlines()
            for i in lignes[1:]:
                target[i.split(";")[0]] = i.split(";")[1].strip()


        label = [] #read each line and add new label in label
        for i in line[1:]:
            if i[0] != '\n':
                if i[0] not in label and i[0] in list(target.keys()):
                    label.append(i[0])


        data = [] #for each label find wavelenght and values
        wl = []
        for i in label:
            tmp_wl = []
            tmp_data = []
            for j in line[1:]:
                if i == j[0] and j[0] in list(target.keys()) and 175 <= float(j[1]) <= 300: # Sélection de la plage de longueur d'onde
                    tmp_wl.append(float(j[1]))
                    tmp_data.append(float(j[2]))
            wl.append(tmp_wl)
            data.append(tmp_data)


    n_wl = []
    n_data = []
    for i in range(len(wl)):
        if wl[i][0] != max(wl[i]):
            tmp_n_wl = []
            tmp_n_data = []
            for j in range(len(wl[i])):
                tmp_n_wl.append(wl[i][-(j+1)])
                tmp_n_data.append(data[i][-(j+1)])
        else:
            tmp_n_wl = []
            tmp_n_data = []
            for j in range(len(wl[i])):
                tmp_n_wl.append(wl[i][j])
                tmp_n_data.append(data[i][j])
        n_wl.append(tmp_n_wl)
        n_data.append(tmp_n_data)


    print(len(n_data), len (label))
    return n_wl, n_data, label, target
    
def class_data(n_wl, n_data, n_label, clas, target):
    """Select wavelength and data of a class.

    Here we select the data for only one class.

    Parameters
    ----------
    n_wl : list
        A list of list of float where each list correspond to the wavelegnth
        of a spectrum.
    n_data : list
        A list of list of float where each list correspond to the value of a
        spectrum at each wevelength, must be the same size as n_wl.
    label : list
        A list of string containing the name of each spectra.
    clas : string
        The class you want to iterate on.
    target : dict
        A dictionnary associating each spectra label to its class.

    Returns
    -------
    r_data : list
        A list of list of float where each list correspond to the value of a
        spectrum at each wevelength, containing only the spectra from a class.
    r_target : dict
        A dictionnary associating each spectra label to its class with only
        the spectra from one class.
    r_label : list
        A list of string containing the name of each spectra for one class.
    """
    n_target = {}
    for i in list(target.keys()):
        if i in n_label:
            n_target[i] = target[i]


    for i in range(len(n_wl)):
        if n_wl[i][0] != 300:
            tmp_n_wl = []
            tmp_n_data = []
            for j in range(len(n_wl[i])):
                tmp_n_wl.append(n_wl[i][-(j+1)])
                tmp_n_data.append(n_data[i][-(j+1)])
        else:
            tmp_n_wl = []
            tmp_n_data = []
            for j in range(len(n_wl[i])):
                tmp_n_wl.append(n_wl[i][j])
                tmp_n_data.append(n_data[i][j])
        n_wl[i] = tmp_n_wl
        n_data[i] = tmp_n_data


    r_data = []
    r_target = []
    r_label = []
    for i in range(len(n_label)):
        if n_target[n_label[i]] in [clas]:
            r_data.append(n_data[i])
            r_target.append(n_target[n_label[i]])
            r_label.append(n_label[i])
    
    return r_data, r_target, r_label


def SVD_on_class(r_data, clas, data, label, wl, invert):
    """
    Performs Singular Value Decomposition (SVD) on a specified class.

    Parameters
    ----------
    r_data : list of list of float
        Data from the specified class.
    clas : str
        Class to process.
    data : list of list of float
        Complete dataset.
    label : list of str
        Names of the spectra.
    wl : list of float
        List of wavelengths.
    invert : bool
        If True, corrects for an inverted spectrum.

    Returns
    -------
    basis : list of numpy.ndarray
        Basis vectors from the SVD.
    df_c : pandas.DataFrame
        Correlation coefficients for the basis vectors and dataset.
    df_p : pandas.DataFrame
        P-values for the correlations.
    """
    norm_r_data = []
    norm_data = []
    for i in r_data:
        value = []
        for j in i:
            value.append(float((j-np.mean(i))/np.std(i)))
        norm_r_data.append(value)

    for i in data:
        value = []
        for j in i:
            value.append(float((j-np.mean(i))/np.std(i)))
        norm_data.append(value)
    
    #wl = np.arange(180, 220.5, 0.5)
    plt.figure(figsize=(15, 5))
    for i in range(len(norm_data)):
        plt.plot(wl, norm_data[i], '-', linewidth = 2.0, label = label[i])
    plt.legend(loc = 4)
    plt.show()
    
    
    A = np.array(norm_r_data)
    u, s, vh = np.linalg.svd(np.transpose(A), full_matrices=True)
    S = np.diag(s)
    S = np.zeros((u.shape[0], vh.shape[0])) #66x71 (IR: 200x50)
    s_size = min(u.shape[0], vh.shape[0])
    S[:s_size, :s_size] = np.diag(s)
    print("A shape :  ", A.shape[0], "x", A.shape[1])
    print("U shape :  ", u.shape[0], "x", u.shape[1])
    print("S shape :  ", S.shape[0], "x", S.shape[1])
    print("Vh shape : ", vh.shape[0], "x", vh.shape[1])
    print("A = u*S*vh ")

    plt.figure(figsize=(15, 5))
    for i in range(2):
        q = u[:, i]
        plt.plot(wl, q, '-', linewidth = 2.0, label = 'Basis #' + str(i) + 'Coeff = ' + "{:.2f}".format(S[i,i]))
    plt.legend(loc = 4)
    plt.title("Without tempscan")
    plt.savefig("./img/"+clas+".png")
    
    basis = [u[:, 0]]
    if invert:
        for i in range(len(basis)):
            basis[i] = -basis[i]
    with open("./Basis_" + clas + ".csv", "w") as filout:
        filout.write("Basis " + clas)
        for i in basis:
            for j in i:
                filout.write(";" + str(j))
    df_c,df_p = cor_on_range(basis, norm_data, label, wl, 1)
    with open("./" + clas + ".csv", "w") as filout:
        filout.write(df_c.to_csv())
    
    return basis, df_c, df_p
    #return basis,norm_data,label,wl
    

def MI_on_class(clas, basis, data, label, wl):
    """
    Calculates normalized mutual information (NMI) between the reference and all spectra.

    Parameters
    ----------
    clas : str
        Class to process.
    basis : list of numpy.ndarray
        Reference basis vectors.
    data : list of list of float
        Complete dataset.
    label : list of str
        Names of the spectra.
    wl : list of float
        List of wavelengths.

    Returns
    -------
    MIs : list of float
        NMI values for all spectra in the dataset.
    """
    norm_data = []
    norm_basis = []
    for i in data:
        value = []
        for j in i:
            value.append(float((j - np.mean(i)) / np.std(i)))
        norm_data.append(value)
    value = []
    for i in basis:
        for j in i:
            value.append(float((j - np.mean(i)) / np.std(i)))
    norm_basis.append(value)
    
    dis_data = []
    for i in norm_data:
        dis = []
        for j in i:
            value = int(round(j, 0))
            dis.append(value)
        dis.reverse()
        dis_data.append(list(dis))


    dis_basis = []
    for i in norm_basis:
        dis = []
        for j in i:
            value = int(round(j, 0))
            dis.append(value)
        dis_basis.append(list(dis))

    plt.figure(figsize=(15, 5))
    for i in range(len(dis_data)):
        plt.plot(wl, dis_data[i], '-', linewidth = 2.0, label = label[i])
    plt.plot(wl,dis_basis[0], '-', linewidth = 4.0, label = "Reference")
    plt.legend(loc = 4)
    plt.show()

    MIs = []
    for i in dis_data:
        MIs.append(calcMI(dis_basis[0], i))

    return MIs


def write_csv(clas, MIs, df_c, label):
    """
    Writes the results to a CSV file.

    Parameters
    ----------
    clas : str
        Class being processed.
    MIs : list of float
        Normalized mutual information values.
    df_c : pandas.DataFrame
        Correlation coefficients for the basis vectors and dataset.
    label : list of str
        Names of the spectra.

    Returns
    -------
    None
    """
    with open(clas + ".csv", "w") as filout:
        filout.write("Spectra;Corr;MI;Corr x MI\n")
        for i in range(len(label)):
            filout.write(label[i] + ";" + str(df_c[i]) + ";" + str(MIs[i]) + ";" + str(df_c[i]*MIs[i]) + "\n")


def cor_on_range(data1, data2, label, interval, number):
    """
    Computes correlation between reference and test datasets.

    Parameters
    ----------
    data1 : list of list of float
        Reference dataset.
    data2 : list of list of float
        Test dataset.
    label : list of str
        Names of the spectra in the test dataset.
    interval : list of float
        List of wavelength intervals.
    number : int
        Number of reference spectra to test.

    Returns
    -------
    df_c : pandas.DataFrame
        Correlation coefficients.
    df_p : pandas.DataFrame
        P-values of the correlations.
    """
    lst_index = []
    for i in range(number):
        lst_index.append('Basis # ' + str(i))

    df1 = pd.DataFrame(index = lst_index, columns = interval)
    k = 0
    for i in lst_index:
        k2=0
        for j in interval:
            df1[j][i] = data1[k][k2]
            k2 += 1
        k += 1

    lst_index = []
    for i in range(len(data2)):
        lst_index.append(label[i])

    df2 = pd.DataFrame(index = lst_index, columns = interval)
    k = 0
    for i in lst_index:
        k2 = 0
        for j in interval:
            df2[j][i] = data2[k][k2]
            k2+=1
        k+=1

    df_c,df_p = corr_2_df(df1,df2)
    return df_c,df_p


def corr_2_df(df1, df2):
    """
    Computes the correlation between two dataframes.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame containing reference spectra data.
    df2 : pandas.DataFrame
        DataFrame containing test spectra data.

    Returns
    -------
    df_c : pandas.DataFrame
        Correlation coefficients.
    df_p : pandas.DataFrame
        P-values of the correlations.
    """
    df_c = pd.DataFrame(index=df2.index.values, columns=df1.index.values)
    df_p = pd.DataFrame(index=df2.index.values, columns=df1.index.values)
    for i in df1.itertuples():
        for j in df2.itertuples():
            df_c[i[0]][j[0]], df_p[i[0]][j[0]] = st.pearsonr(i[1:], j[1:])
    return df_c, df_p

if __name__ == '__main__':
    file = sys.argv[1]
    reffile = sys.argv[2]
    outpath = sys.argv[3]
    col_or_lin = sys.argv[4]
    clas = sys.argv[5]
    test = sys.argv[6]
    invert = sys.argv[7]
    main(file, reffile, outpath, col_or_lin, clas, test, invert)
