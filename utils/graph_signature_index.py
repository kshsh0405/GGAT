#argument: Planetoid & WebKB & WikipediaNetwork graph dataset (for node classification datasets with name variable)
#Planetoid: undirected graph without self-loops. Each edge doubly counted.
#WebKB & WikipediaNetwork: directed graph. Some self-loops added.
class NodeC_Graph_Signature:

    #initialize
    def __init__(self, data_x, data_y, data_edge_index, token=False, sl_token=False):
        self.graph_data_x = data_x
        self.graph_data_y = data_y
        self.graph_edge_index = data_edge_index

        self.symmetrize = token
        self.selfloop = sl_token

        #compile dimension of node labels

        max_label = np.unique(data_y)
        self.label_dim = len(max_label)
        self.label = max_label

        #construct a library of node labels

        library_list = list(np.unique(data_y))
        #library_list = []
        #for i in range(0,self.label_dim):
        #    library_list += [i]
        self.library = torch.tensor(library_list)
        print("Graph Dataset Compiled:")

    def graph_main(self):
        signature_list = [0,0,0]

        start_time = time.time()
        graph_edge_source, graph_edge_target = self.graph_preprocess()
        end_time = time.time() - start_time
        print(f"Compile time: {end_time}")
        print(" ")

        #edge homophily index
        start_time = time.time()
        graph_edge_hom = self.edge_homophily(graph_edge_source, graph_edge_target)
        end_time = time.time() - start_time
        print(f"Compile time: {end_time}")
        print(" ")

        #node homophily index
        #start_time = time.time()
        #graph_node_hom = self.node_homophily(graph_edge_source, graph_edge_target)
        #end_time = time.time() - start_time
        #print(f"Compile time: {end_time}")
        #print(" ")

        #normalized edge homophily index
        #start_time = time.time()
        #graph_norm_edge_hom = self.norm_edge_homophily(graph_edge_source, graph_edge_target)
        #end_time = time.time() - start_time
        #print(f"Compile time: {end_time}")
        #print(" ")

        #graph signature matrix
        start_time = time.time()
        graph_sig_data, graph_eig_list, graph_eig_vect = self.graph_signature_index(graph_edge_source, graph_edge_target)
        for i in graph_eig_list:
            if i > 1e-5:
                signature_list[0] += 1
            elif i < -1e-5:
                signature_list[2] += 1
            else:
                signature_list[1] += 1
        end_time = time.time() - start_time
        print("Graph Signature Index: ", signature_list)
        print(f"Compile time: {end_time}")
        print(" ")

        #return graph_sig_data, graph_eig_list, graph_eig_vect, signature_list, graph_edge_hom, graph_node_hom, graph_norm_edge_hom
        return graph_sig_data, graph_eig_list, graph_eig_vect, signature_list, graph_edge_hom

    def graph_supple(self):
        start_time = time.time()
        graph_edge_source, graph_edge_target = self.graph_preprocess()
        end_time = time.time() - start_time
        print(f"Compile time: {end_time}")
        print(" ")

        #node homophily index
        start_time = time.time()
        graph_node_hom = self.node_homophily(graph_edge_source, graph_edge_target)
        end_time = time.time() - start_time
        print(f"Compile time: {end_time}")
        print(" ")

        #normalized edge homophily index
        start_time = time.time()
        graph_norm_edge_hom = self.norm_edge_homophily(graph_edge_source, graph_edge_target)
        end_time = time.time() - start_time
        print(f"Compile time: {end_time}")
        print(" ")

        return graph_node_hom, graph_norm_edge_hom

    def graph_preprocess(self):
        #import ith graph dataset
        graph_node = self.graph_data_y
        graph_node_num = len(graph_node)
        graph_node_label = np.unique(graph_node)

        #construct a directed graph G' from G. Specified by arguments "token" (symmetrize) and "sl_token" (self loop)
        self_loop = torch.tensor([i for i in range(0,graph_node_num)])
        if self.symmetrize and self.selfloop:
            graph_edge_source = torch.cat(((self.graph_edge_index)[0],(self.graph_edge_index)[1],self_loop))
            graph_edge_target = torch.cat(((self.graph_edge_index)[1],(self.graph_edge_index)[0],self_loop))
        elif not self.symmetrize and self.selfloop:
            graph_edge_source = torch.cat(((self.graph_edge_index)[0],self_loop))
            graph_edge_target = torch.cat(((self.graph_edge_index)[1],self_loop))
        elif self.symmetrize and not self.selfloop:
            graph_edge_source = torch.cat(((self.graph_edge_index)[0],(self.graph_edge_index)[1]))
            graph_edge_target = torch.cat(((self.graph_edge_index)[1],(self.graph_edge_index)[0]))
        else:
            graph_edge_source = (self.graph_edge_index)[0]
            graph_edge_target = (self.graph_edge_index)[1]

        print("Graph Preprocessed:")
        return graph_edge_source, graph_edge_target

    #graph node label dictionary
    def graph_node_dict_preprocess(self):
        graph_node = self.graph_data_y
        graph_node_dict = [0]*(self.label_dim)
        i = 0
        for j in self.label:
            local_label = (graph_node == j).nonzero(as_tuple=True)[0]
            graph_node_dict[i] += len(local_label)
            i += 1
        return graph_node_dict

    #graph edge label dictionary
    def graph_edge_dict_preprocess(self, graph_edge_source, graph_edge_target):
        graph_node = self.graph_data_y
        graph_edge_dict = [[float(0)]*self.label_dim]*self.label_dim
        edge_list = [(m,n) for m in self.label for n in self.label]
        graph_edge_dict = torch.tensor(graph_edge_dict)
        for (j,k) in edge_list:
            loc_s_label = (graph_node[graph_edge_source] == int(j)).nonzero(as_tuple=True)[0]
            loc_t_label = (graph_node[graph_edge_target] == int(k)).nonzero(as_tuple=True)[0]
            new_st_label = np.intersect1d(loc_s_label, loc_t_label)
            graph_edge_dict[j,k] += len(new_st_label)

        return graph_edge_dict

    #currently implemented for undirected graphs without self loops!!
    #WebKB stores directed graphs as G' without self loops.
    #We will allow users to choose whether one would symmetrize the graph G' (each directed edge has a corresponding reverse-directional edge)
    def graph_signature(self, graph_edge_source, graph_edge_target):
        graph_node = self.graph_data_y
        graph_node_num = len(graph_node)
        graph_edge_num = len(graph_edge_source)

        #import ith graph dataset
        graph_node = self.graph_data_y
        graph_node_num = len(graph_node)
        graph_node_dict = self.graph_node_dict_preprocess()
        graph_edge_dict = self.graph_edge_dict_preprocess(graph_edge_source, graph_edge_target)

        #compute graph signature matrix
        graph_sig_mat = graph_edge_dict
        for j in range(0,self.label_dim):
            for k in range(0,self.label_dim):
                entry = float(graph_sig_mat[j][k])
                if entry > 0:
                    normalize = float((graph_node_dict[j])*(graph_node_dict[k]))
                    graph_sig_mat[j][k] = entry/normalize
        graph_sig_mat = 1/2*(graph_sig_mat + torch.transpose(graph_sig_mat,0,1))

        print("Signature_matrix Computed:")
        print(graph_sig_mat)
        return graph_sig_mat

    #compute eigenvalues of graph signature matrices
    #If the graph signature matrix is not symmetric, symmetrize the matrix by 1/2*(S + S^T)
    def graph_signature_index(self,graph_edge_source, graph_edge_target):
        from torch import linalg as LA
        graph_sig_mat_data = self.graph_signature(graph_edge_source, graph_edge_target)
        eig_list = []
        eig_vect = []

        sig_mat = graph_sig_mat_data

            #compute eigenvalues & corresponding eigenvectors
            #The eigenvectors with zero eigenvalues are appended after those with non-zero eigenvalues.
        eig_l, eig_q = LA.eig(sig_mat)

            #reorder eigenvalues in ascending order except zero eigenvalues and, accordingly, their eigenvectors
        orig = (eig_l.real).ne(0)
        sort_eig_l = (eig_l.real)[orig]
        sorted, indices = torch.sort(sort_eig_l)

        basis_matrix = torch.tensor([[float(0)]*self.label_dim]*self.label_dim)
        for i in range(0,self.label_dim):
            for j in range(0,self.label_dim):
                if i < len(indices):
                    if j == indices[i]:
                        basis_matrix[i,j] += 1
                else:
                    basis_matrix[i,i] += 1
                    break
        mid_eig_l = torch.matmul(eig_l.real, basis_matrix)
        mid_eig_q = torch.matmul(eig_q.real, basis_matrix)

            #record which node labels appear in the graph.
        proj_mat = (torch.sum(sig_mat, axis=0)).gt(0)
        rk = (proj_mat == True).nonzero(as_tuple=True)[0]
        ker = (proj_mat == False).nonzero(as_tuple=True)[0]

            #construct a change of basis matrix
        basis_matrix = torch.tensor([[float(0)]*self.label_dim]*self.label_dim)
        for i in range(0,self.label_dim):
            for j in range(0,self.label_dim):
                if i < len(rk):
                    if j == rk[i]:
                        basis_matrix[i,j] += 1
                else:
                    if j == ker[i-len(rk)]:
                        basis_matrix[i,j] += 1

            #change of coordinates
        new_eig_l = torch.matmul(mid_eig_l, basis_matrix)
        new_eig_q = torch.matmul(mid_eig_q, basis_matrix)

            #obtain eigenvalues respecting the occurrences of node labels
        eig_list = new_eig_l
        eig_vect = new_eig_q

        print("Signature_matrix_eigenvalues Computed:")
        print(eig_list)
        return graph_sig_mat_data, eig_list, eig_vect

    #edge homophily index from Zhu et al. (2021) H2GCN:
    #"Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
    #NeurIPS 2021
    def edge_homophily(self, graph_edge_source, graph_edge_target):
        graph_edge_hom = 0
        graph_edge_same = 0
        graph_edge_dict = self.graph_edge_dict_preprocess(graph_edge_source, graph_edge_target)

        for k in range(0,len(graph_edge_dict)):
            graph_edge_same += graph_edge_dict[k][k]
        graph_edge_hom = float(graph_edge_same / len(graph_edge_source))
        print("Edge homophily index Computed:")
        print("h_E: ", graph_edge_hom)
        return graph_edge_hom

    #node homophily index from Pei et al. (2020) Geom-GCN:
    #"Geom-GCN: Geometric Graph Convolutional Networks"
    #ICLR 2020
    def node_homophily(self, graph_edge_source, graph_edge_target):
        graph_node_hom = 0
        graph_node_same = 0
        graph_node = self.graph_data_y

        for k in range(0,len(graph_node)):
            local_target = graph_edge_target[(graph_edge_source == k).nonzero(as_tuple=True)[0]]
            local_target_label = graph_node[local_target]
            true_label = graph_node[k]
            local_same = len((local_target_label == true_label).nonzero(as_tuple=True)[0]) + 1e-8
            local_deg = len(local_target) + 1e-8
            graph_node_same += float(local_same / local_deg)

        graph_node_hom = float(graph_node_same / len(graph_node))
        print("Node homophily index Computed:")
        print("h_V: ", graph_node_hom)
        return graph_node_hom

    #normalized edge homophily index from Lim et al. (2021) LinkX:
    #"Large Scale Learning on Non-homophilous Graphs: New Benchmarks and Strong Simple Methods"
    def norm_edge_homophily(self, graph_edge_source, graph_edge_target):
        graph_norm_edge_hom = 0
        graph_node = self.graph_data_y

        for l in range(0,self.label_dim):
            local_node_with_label = (graph_node == l).nonzero(as_tuple=True)[0]
            local_node_with_label_num = len(local_node_with_label)
            if local_node_with_label_num == 0:
                continue
            global_same = 0
            global_deg = 0
            for k in local_node_with_label:
                local_target = graph_edge_target[(graph_edge_source == k).nonzero(as_tuple=True)[0]]
                local_target_label = graph_node[local_target]
                local_same = len((local_target_label == l).nonzero(as_tuple=True)[0]) + 1e-8
                local_deg = len(local_target) + 1e-8
                global_same += local_same
                global_deg += local_deg
            global_entry = float(global_same / global_deg - local_node_with_label_num / len(graph_node))
            if global_entry > 0:
                graph_norm_edge_hom += global_entry
        graph_norm_edge_hom = float(graph_norm_edge_hom / (self.label_dim-1))
        print("Normalized homophily index Computed:")
        print("h^_E: ", graph_norm_edge_hom)
        return graph_norm_edge_hom


def GraphS_compute(data_x, data_y, data_edge):
    data_sig = NodeC_Graph_Signature(data_x, data_y, data_edge, token=False, sl_token=False)
    data_sig_mat, data_eig_list, data_eig_vect, data_eig_sig, data_edge_hom = data_sig.graph_main()
    return data_sig_mat, data_eig_list, data_eig_vect, data_eig_sig, data_edge_hom

def Graph_supple_compute(data_x, data_y, data_edge, token_input=False, sl_token_input=False):
    data_supple = NodeC_Graph_Signature(data_x, data_y, data_edge, token=token_input, sl_token=sl_token_input)
    data_supple_mat, data_supple_eig_list, data_supple_eig_vect, data_supple_eig_sig, data_supple_edge_hom = data_supple.graph_main()
    data_supple_node_hom, data_supple_norm_edge_hom = data_supple.graph_supple()
    return data_supple_node_hom, data_supple_norm_edge_hom
