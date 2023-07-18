import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math


def PushMatrixNaive(matrix, wdm_ch_num):
    core_num = math.ceil(matrix.shape[0]/wdm_ch_num)
    
    for c in range(core_num):
        partial_pushed_matrix = np.zeros((min(wdm_ch_num, matrix.shape[0]-c*wdm_ch_num), matrix.shape[1]))
        for i in range(min(wdm_ch_num, matrix.shape[0]-c*wdm_ch_num)):
            row_vec = []
            for x in matrix[i]:
                if x != 0:
                    row_vec.append(x)
            new_row_vec = [0 for j in range(matrix.shape[1]-len(row_vec))]
            partial_pushed_matrix[i, :] = row_vec + new_row_vec
        
        if c == 0:
            pushed_matrix = partial_pushed_matrix
        else:
            np.concatenate((pushed_matrix, partial_pushed_matrix), axis=0)

    return pushed_matrix


def PushMatrixGentle(matrix, wdm_ch_num):
    core_num = math.ceil(matrix.shape[0]/wdm_ch_num)
    
    for c in range(core_num):
        partial_pushed_matrix = np.copy(matrix[c*wdm_ch_num: c*wdm_ch_num+min(wdm_ch_num, matrix.shape[0]-c*wdm_ch_num), :])
        empty_column_index = []
        for j in range(matrix.shape[1]):
            column = list(matrix[:, j])
            if sum(column) == 0:
                empty_column_index.append(j)
        partial_pushed_matrix = np.delete(partial_pushed_matrix, empty_column_index, axis=1)
        partial_pushed_matrix = np.concatenate((partial_pushed_matrix, np.zeros((min(wdm_ch_num, matrix.shape[0]-c*wdm_ch_num), matrix.shape[1]-partial_pushed_matrix.shape[1]))), axis=1)

        if c == 0:
            pushed_matrix = partial_pushed_matrix
        else:
            np.concatenate((pushed_matrix, partial_pushed_matrix), axis=0)
    
    return pushed_matrix
        

def Plot2DMatrix(matrix, matrix_name, color_map):
    max_value = max(matrix.max(), abs(matrix.min()))
    plt.matshow(matrix, cmap=color_map)
    plt.colorbar()
    plt.clim(-max_value, max_value)
    plt.title(matrix_name)
    plt.show()

    count_zero = 0
    for i in tqdm(range(matrix.shape[0])):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                count_zero += 1
    print("There are totally", count_zero, "zero elements in this matrix, percentage is", count_zero/(matrix.shape[0]*matrix.shape[1]))


## plot the different vector sizes of a matrix
def PlotVectorSizes(weight_matrix, pruned_matrix, matrix_name, plot_two_matrces=False):
    row_nonzero_set = []
    for i in tqdm(range(weight_matrix.shape[0])):
        row_nonzero = 0
        for j in range(weight_matrix.shape[1]):
            if weight_matrix[i,j] != 0:
                row_nonzero += 1
        row_nonzero_set.append(row_nonzero)
    
    pruned_row_nonzero_set = []
    if plot_two_matrces:
        for i in tqdm(range(pruned_matrix.shape[0])):
            pruned_row_nonzero = 0
            for j in range(pruned_matrix.shape[1]):
                if pruned_matrix[i,j] != 0:
                    pruned_row_nonzero += 1
            pruned_row_nonzero_set.append(pruned_row_nonzero)

    fig, axs = plt.subplots(1, 3, figsize=(18,3), sharey=False, tight_layout=True)

    axs[0].scatter(list(range(weight_matrix.shape[0])), row_nonzero_set, marker='o', alpha=0.5, label="initial matrix")
    if plot_two_matrces:
        axs[0].scatter(list(range(pruned_matrix.shape[0])), pruned_row_nonzero_set, marker='o', alpha=0.5, label="pruned matrix")
    axs[0].legend()
    axs[0].set_xlabel("Row index of the matrix")
    axs[0].set_ylabel("Number of nonzero elements")

    cdf_row_nonzero_set = []
    sorted_row_nonzero_set = row_nonzero_set.copy()
    sorted_row_nonzero_set.sort()
    for i in range(len(sorted_row_nonzero_set)):
        cdf_row_nonzero_set.append(i/len(sorted_row_nonzero_set))

    if plot_two_matrces:
        cdf_pruned_row_nonzero_set = []
        sorted_pruned_row_nonzero_set = pruned_row_nonzero_set.copy()
        sorted_pruned_row_nonzero_set.sort()
        for i in range(len(sorted_pruned_row_nonzero_set)):
            cdf_pruned_row_nonzero_set.append(i/len(sorted_pruned_row_nonzero_set))
        axs[1].bar(list(range(len(sorted_pruned_row_nonzero_set))), sorted_pruned_row_nonzero_set)
    else:
        axs[1].bar(list(range(len(sorted_row_nonzero_set))), sorted_row_nonzero_set)
    axs[1].set_xlabel("Sorted Row index of the matrix")
    axs[1].set_ylabel("Number of nonzero elements")
    
    axs[2].plot(sorted_row_nonzero_set, cdf_row_nonzero_set, marker=".", alpha=0.5, label="initial matrix")
    if plot_two_matrces:
        axs[2].plot(sorted_pruned_row_nonzero_set, cdf_pruned_row_nonzero_set, marker=".", alpha=0.5, label="pruned matrix")
    axs[2].legend()
    axs[2].set_xlabel("Number of nonzero elements")
    axs[2].set_ylabel("CDF")

    fig.suptitle(matrix_name)
    
    return row_nonzero_set, pruned_row_nonzero_set


## plot the distribution of all elements in the matrix (flattened). This is very slow
def PlotMatrixAllDistribution(matrix):
    max_value = max(matrix.max(), abs(matrix.min()))
    sorted_matrix = list(matrix.flatten())
    sorted_matrix.sort()
    cdf_matrix = [x/len(sorted_matrix) for x in range(0, len(sorted_matrix))]

    distinct_values = [sorted_matrix[0]]
    for i in tqdm(range(len(sorted_matrix)-1)):
        if sorted_matrix[i+1] > sorted_matrix[i]:
            distinct_values.append(sorted_matrix[i+1])

    fig, axs = plt.subplots(1, 2, figsize=(12,3), sharey=False, tight_layout=True)
    axs[0].plot(sorted_matrix, cdf_matrix, marker=".", alpha=0.5)
    axs[0].set_xlim(left=-max_value, right=max_value)
    axs[0].set_xlabel("Matrix's element number")
    axs[0].set_ylabel("CDF")

    axs[1].hist(matrix.flatten(), bins=len(distinct_values), density=True, facecolor="green", alpha=0.8)
    # axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=len(matrix.flatten())//10000))
    axs[1].set_xlim(left=-max_value, right=max_value)
    axs[1].set_xlabel("Matrix's element number")
    axs[1].set_ylabel("Probability")


def main():
    pass


if __name__ == "__main__":
    main()