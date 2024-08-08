import pickle
from torch_geometric.data import HeteroData


def construct_graph(path_graph, device):

    # load data
    graph_data = pickle.load(open(path_graph, 'rb'))
    # construct data
    graph = HeteroData()
    graph['cell'].x = graph_data['node_cell']
    graph['tract'].x = graph_data['node_tract']
    graph['cell', 'in', 'tract'].edge_index = graph_data['edge_in']
    graph['cell', 'near', 'cell'].edge_index = graph_data['edge_near_cell']
    graph['cell', 'flow', 'cell'].edge_index = graph_data['edge_flow_cell_idx']
    graph['cell', 'flow', 'cell'].edge_attr = graph_data['edge_flow_cell_attr']
    graph['tract', 'near', 'tract'].edge_index = graph_data['edge_near_tract']
    graph['tract', 'flow', 'tract'].edge_index = graph_data['edge_flow_tract_idx']
    graph['tract', 'flow', 'tract'].edge_attr = graph_data['edge_flow_tract_attr']
    # normalize node/edge features
    for key, values in graph.x_dict.items():
        graph[key].x = MinMaxScaler(values)
    for key, values in graph.edge_attr_dict.items():
        graph[key].edge_attr = MinMaxScaler(values)

    print('==== graph info:')
    print(graph)
    pickle.dump(graph, open('./data/graph.pkl', 'wb'))
    print('graph saved!')

    return graph.to(device)



def MinMaxScaler(data):
    row_max = data.max(dim=0)[0]
    row_min = data.min(dim=0)[0]
    updated = (data - row_min) / (row_max - row_min)
    updated = updated.nan_to_num(0)
    return updated


