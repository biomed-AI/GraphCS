import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_bar_chart2_for_graph_ablation(mat, x_label, save_name="plot6", a_SD=None):
    '''
    :param mat:data
    :param x_label: label
    :param save_name:
    :return:
    '''
    fig =plt.figure(figsize=(10, 7))
    weight = "bold"
    colors = ['#700000',
              "#800000",
              '#900000',
              '#a00000',
              '#b00000',
              "#c00000",
              '#d00000',
              "#e00000",
              '#f00000',
              ]

    colors_dict = {"scmap": "#8C564B",
                   "singleCellNet": "#92D050",
                   "Seurat-CCA": "#00B0F0",
                   "Seurat-PCA": "#F59D56",
                   "SingleR": "#BFBFBF",
                   "scGCN": "#FFD965",
                   "GBPCs": "#C00000",
                   "CHETAH": "#CC00FF",
                   "scPred": "#FF6699"}


    # sort_index = np.argsort(mat)
    # x_label = np.asarray(x_label)[sort_index]
    # mat = np.sort(mat)

    # normalize
    norm_values = 0.4 + 0.5 * (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    map_vir = cm.get_cmap(name='Blues')
    colors = map_vir(norm_values)

    # plt.title(, fontsize=30)

    if a_SD:
        plt.bar(x_label, mat, color=colors, yerr = a_SD)
    else:
        plt.bar(x_label, mat, color=colors)

    plt.xticks(fontsize=28, weight=weight)  
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=28,weight=weight)  
    plt.ylabel("Accuracy", fontsize=28, fontweight=weight)
    fig.autofmt_xdate(rotation=45)

    # note for bar
    # for x, y in enumerate(mat):
    #     plt.text(x + 0.05, y + 0.02, '%.2f' % y, ha='center', va='bottom', fontsize=23)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.vlines(3, 0, 1.0, colors="r", linestyles="dashed")

    #ax.set_ylim(bottom=0.5)
    if weight:
        ax.spines['bottom'].set_linewidth(2)  
        ax.spines['left'].set_linewidth(2)  

    #
    plt.savefig('{}.png'.format(save_name), format='png', dpi=800, bbox_inches='tight')
    plt.show()






results = open("result.out").read().splitlines() 

accs={}

for each_result in results:

    method=each_result.split("!")[0]
    method_acc=float(each_result.split(":")[-1])
    
    if method in accs.keys():
        accs[method].append(method_acc)
    else: 
        accs[method]=[method_acc]

for method_name, acc_list in accs.items():
    print(method_name, " mean: ", np.mean(np.array(acc_list), axis=0), " std: ",np.std(np.array(acc_list), axis=0))
       

mat=np.asarray([

np.mean(np.array(accs['GraphCS'])),
np.mean(np.array(accs['VAT-'])),
np.mean(np.array(accs['GBP-'])),
0,
np.mean(np.array(accs['conos'])),
np.mean(np.array(accs['ccamnn'])),
np.mean(np.array(accs['sklearnknn'])),
np.mean(np.array(accs['cosine'])),
np.mean(np.array(accs['umap'])),

np.mean(np.array(accs['annoy'])),

])

a_SD=[

np.std(np.array(accs['GraphCS'])),
np.std(np.array(accs['VAT-'])),
np.std(np.array(accs['GBP-'])),
0,
np.std(np.array(accs['conos'])),
np.std(np.array(accs['ccamnn'])),
np.std(np.array(accs['sklearnknn'])),
np.std(np.array(accs['cosine'])),
np.std(np.array(accs['umap'])),
np.std(np.array(accs['annoy'])),


]

methods = [
          "GraphCS",
           "-VAT",
           "-GBP",
               "",
           "Conos",
           "CCA-MNN",
              "KNN",
           "Cosine",
            "UMAP",
           "Annoy",
]



plot_bar_chart2_for_graph_ablation(mat, methods, save_name="plot601", a_SD=a_SD)



