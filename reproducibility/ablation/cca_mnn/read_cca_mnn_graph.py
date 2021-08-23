import pandas as pd
import numpy as np

def get_cca_graph(datastr):

    id_graph1 = pd.read_csv('{}/inter_graph.csv'.format("./input/"+datastr),
                            index_col=0,
                            sep=',').values
    id_graph2 = pd.read_csv('{}/intra_graph.csv'.format("./input/"+datastr), sep=',', index_col=0).values
    label1 = pd.read_csv('{}/Label1.csv'.format("./input/"+datastr), header=0, index_col=False, sep=',').values.flatten()
    label2 = pd.read_csv('{}/Label2.csv'.format("./input/"+datastr), header=0, index_col=False, sep=',').values.flatten()
    len_label1=len(label1)
    len_label2=len(label2)

    id_graph2=id_graph2+len_label1
    id_graph1[:,1]=id_graph1[:,1]+len_label1

    f = open("../data/" + datastr + "_ccamnn.txt", 'w')
    f.write('{}\n'.format(len_label1+len_label2))

    graph_matrix=np.vstack((id_graph2, id_graph1))
    rows, cols=graph_matrix.shape

    for i in range(rows):
        f.write('{} {}\n'.format(graph_matrix[i][0], graph_matrix[i][1]))

    f.close()


name_list=[

'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human',
'Baron_human_Baron_mouse'

]
for filename in name_list:
    get_cca_graph(filename)

