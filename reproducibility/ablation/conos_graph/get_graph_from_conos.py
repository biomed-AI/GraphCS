import scanpy as sc

DATA_PATH = "./conos_origin_graph/"


def save_graph_from_cons_add_self_edge(data_name=None, train_index=None):

    graph_conn_mtx = sc.read(DATA_PATH +data_name+ "/graph_connectivities.mtx").X.A
    from collections import defaultdict
    n_count = graph_conn_mtx.shape[0]
    f = open("../data/" + data_name + "_conos.txt", 'w')
    f.write('{}\n'.format(n_count))
    adj = defaultdict(list)  # default value of int is 0
    for i, row in enumerate(graph_conn_mtx):
        for j, adjacent in enumerate(row):
            if adjacent and adjacent>0:
                adj[i].append(j)
                f.write('{} {}\n'.format(i, j))

        if adj[i].__len__ == 0:
            adj[i] = []
    f.close()
    return adj




if __name__=="__main__":
    
    name_list=[
     'Baron_mouse_combination',
      'Baron_mouse_segerstolpe',
     'Baron_mouse_Baron_human',
     'Baron_human_Baron_mouse'
  
     ]

    for file_name in name_list:
        save_graph_from_cons_add_self_edge(data_name=file_name)



