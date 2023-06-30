import numpy as np
import random
import matplotlib.pyplot as plt

#load the data adn spreat the target into y and the featuers into X
def data_processing():
    X=[]
    y=[]
    words=["animals","countries","fruits","veggies"]
    for word in words:
        with open('data/'+word, 'r') as f:
            for line in f:
                values = line.split()
                X.append(np.asarray(values[1:], dtype='float32'))
                y.append(word)
    return X,y

class SOM:
    def __init__(self, input_dim, K, learning_rate=0.5, neighborhood_radius=1.0):
        np.random.seed(0)
        self.k=K
        self.weights = np.random.rand(K, input_dim)  
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius

    def get_best_matching_unit(self, input_vector):
        distance_map = np.linalg.norm(self.weights - input_vector, axis=1)
        return np.argmin(distance_map)

    def update_weights(self, input_vector, bmu_index):
        for i in range(self.weights.shape[0]):
            distance = np.linalg.norm(np.array([i]) - np.array([bmu_index]))
            if distance <= self.neighborhood_radius:
                influence = np.exp(-(distance ** 2) / (2 * (self.neighborhood_radius ** 2)))
                self.weights[i] += self.learning_rate * influence * (input_vector - self.weights[i])

    def train(self, data, epochs=1000):
        for epoch in range(epochs):
            input_vector = random.choice(data)
            bmu_index = self.get_best_matching_unit(input_vector)
            self.update_weights(input_vector, bmu_index)

    #get the cluster of the data
    def cluster(self, data):
        clusters = np.zeros(len(data))
        for i, input_vector in enumerate(data):
            bmu_index = self.get_best_matching_unit(input_vector)
            clusters[i] = bmu_index
        return clusters

    #calculate the macro avrage precision/recall/F-score
    def claculate_P_R_F(self, true_y, predicted_y):
        words=["animals","countries","fruits","veggies"]
        macr_avr_p= macr_avr_r= macr_avr_f=0
        for word in words:
            p= r= f=0
            TP= FP= FN=0
            for w1, w2 in zip(true_y, predicted_y):
                TP += (w1 == w2 == word)
                FP += (w1!= w2 and w2== word)
                FN += (w1!=w2 and w1==word)
            #this condtion to get rid of the cases where a catogry didn't exist at all on the clustered data
            if(TP==0 and FP==0):
                p=0.00000001
            else:
               p = (TP/(TP+FP))
            r = (TP/(TP+FN)) 
            f = ((2*p*r)/(p+r))
            macr_avr_p += p
            macr_avr_r += r
            macr_avr_f += f
        macr_avr_p /= 4
        macr_avr_r /= 4
        macr_avr_f /= 4
        return macr_avr_p, macr_avr_r, macr_avr_f

    #find out the dominant label for each cluster 
    def get_labeled_clusters(self,clusters):
        labeled_clusters=[]
        dominant_label = []
        #fill the dominant_label
        for i in range(self.k):
            d = {'animals': 0, 'countries': 0, 'fruits': 0, 'veggies': 0}
            dominant_label.append(d)

        #cont how many each cluster appears
        for i in range(len(clusters)):
            clust=y[i]
            dominant_label[clusters[i].astype(int)][clust]+=1

        #fill the labeled_clusters with dominant label 
        for i in range(len(clusters)):
            labeled_clusters.append(max(dominant_label[clusters[i].astype(int)], key=lambda k: dominant_label[clusters[i].astype(int)][k]))

        return labeled_clusters



if __name__=="__main__":
    X,y=data_processing()
    X=np.array(X)
    y=np.array(y)

    #data info printing
    print("Data information: \n X shape:{0} y shape : {1} Clusters : {2}".format(X.shape,y.shape,np.unique(y)))

    #train the SOM
    input_dim=len(X[0])
    K=int(np.sqrt(5 * np.sqrt(input_dim)))
    som = SOM(input_dim, K)
    som.train(X, epochs=1000)
    clusters = som.cluster(X)
    labeled_clusters=som.get_labeled_clusters(clusters)
    p,r,f=som.claculate_P_R_F(y,labeled_clusters)
    print("\n With {0} clusters the macro avrage precision: {1:.3f} recall: {2:.3f} F-score: {3:.3f}\n".format(K,p,r,f)) 

    #tray different k and calculate the macro avrage precision/recall/F-score
    k_values = range(2, 11)
    labeled_clusters=[]
    macr_avr_p=[]
    macr_avr_r=[]
    macr_avr_f=[]
    for k in k_values:
        som = SOM(input_dim, k)
        som.train(X, epochs=1000)
        clusters = som.cluster(X)
        labeled_clusters=som.get_labeled_clusters(clusters)
        p,r,f=som.claculate_P_R_F(y,labeled_clusters)
        macr_avr_p.append(p)
        macr_avr_r.append(r)
        macr_avr_f.append(f)
        print("for k {0} : cluster/s the macro avrage precision: {1:.3f} recall: {2:.3f} F-score: {3:.3f}".format(k,p,r,f))
        
    #plot the numbdr of cluster VS Metrics
    plt.title('Performance of SOM clustering  VS Number of clusters')
    plt.plot(k_values, macr_avr_p, label='precision Score')
    plt.plot(k_values, macr_avr_r, label='recall Score')
    plt.plot(k_values, macr_avr_f, label='F Score')
    plt.ylabel('Metrics')
    plt.xlabel('Number of clusters (k)')
    plt.legend()
    plt.show()
    
