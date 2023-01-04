import numpy as np
from sklearn.neighbors import NearestNeighbors
from cvxpy import Minimize, Maximize, Problem, Variable


# An object oriented optimization problem.                               
class Edge:                                                          
    """ An undirected, capacity limited edge. """                    
    def __init__(self, capacity, n1, n2, cost) -> None:              
        self.capacity = capacity                                     
        self.cost = cost                                             
        self.flow = Variable(name=f"{n1}_{n2}")                      
                                                                     
    # Connects two nodes via the edge.                               
    def connect(self, in_node, out_node):                            
        in_node.edge_flows.append(-self.flow)                        
        out_node.edge_flows.append(self.flow)                        
                                                                     
    # Returns the edge's internal constraints.                       
    def constraints(self):                                           
        return [0 <= self.flow, self.flow <= self.capacity]          
                                                                     
class Node:                                                          
    """ A node with accumulation. """                                
    def __init__(self, accumulation: float = 0.0) -> None:           
        self.accumulation = accumulation                             
        self.edge_flows = []                                         
                                                                     
    # Returns the node's internal constraints.                       
    def constraints(self):                                           
        return [sum(f for f in self.edge_flows) == self.accumulation]


# Flow optimization
def opt_flow(x0, train_data, model, K, n_neighbors, tau, X_diverse):
    # Predict labels
    labels = model.predict(train_data)
    train_data_ = np.concatenate([x0.reshape(1, -1), train_data])
    labels_ = np.concatenate([model.predict(x0.reshape(1, -1)), labels])

    # Build graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(train_data_)
    distances, indices = nbrs.kneighbors(train_data_)
    graph = nbrs.kneighbors_graph(train_data_)

    node_count = graph.shape[0]                
    nodes = [Node() for i in range(node_count)]

    # Idx for prototypes
    destination = []
    for i in range(X_diverse.shape[0]):
        # idx_l = np.where(train_data_ == X_diverse[i])
        idx_l = np.where(np.prod(train_data_ == X_diverse[i], axis = -1))
        destination.append(idx_l[0][0])                    

    # Constraints for source and sinks
    nodes[0].accumulation = K
    for i in range(len(destination)):                           
        nodes[destination[i]].accumulation = -1                 

    # Constraints for edges
    edges, edges_checked = [], []
    for i in range(graph.shape[0]):
        for j in range(1, indices.shape[1]):
            if (i, indices[i][j]) in edges_checked:
                continue
            dist = tau * distances[i][j] if (labels_[i] == 0 or labels_[j] == 0) else (1 - tau) * distances[i][j]
            cap = 1 if labels_[i] == 0 and labels_[j] == 1 else K

            edges.append(Edge(cap, i, indices[i][j], dist))
            edges[-1].connect(nodes[i], nodes[indices[i][j]])
            edges_checked.append((nodes[i], nodes[indices[i][j]]))

            edges.append(Edge(cap, indices[i][j], i, dist))
            edges[-1].connect(nodes[indices[i][j]], nodes[i])
            edges_checked.append((nodes[indices[i][j]], nodes[i]))
    
    # Construct the problem.                       
    constraints = []                               
    for o in nodes + edges:                        
        constraints += o.constraints()             
    expression = edges[0].cost * edges[0].flow     
    for i in range(1, len(edges)):                 
        expression += edges[i].cost * edges[i].flow
    p = Problem(Minimize(expression), constraints) 
    result = p.solve()                             

    # Results
    res, res_idx = [], []
    for variable in p.variables():
        if variable.value > 0.1:
            i, j = int(variable.name().split('_')[0]), int(variable.name().split('_')[1])
            if (i, j) not in res:
                res.append((i, j))
                if labels_[i] == 0 and labels_[j] == 1:
                    res_idx.append(train_data_[j])
                if labels_[i] == 1 and labels_[j] == 0:
                    res_idx.append(train_data_[i])

    return res_idx[:K], res
