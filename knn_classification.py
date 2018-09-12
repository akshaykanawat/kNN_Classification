import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as ss



def distance(p1,p2):
    """Returns euclidean distance"""
    return np.sqrt(np.sum(np.power(p1-p2,2)))

def majority_vote(points):
    """Returns most common element in the votes"""
    vote_count={}
    for vote in points:
        if vote in vote_count:
            vote_count[vote]+=1
        else:
            vote_count[vote]=1
    winner=[]
    for vote,count in vote_count.items():
        if count==max(vote_count.values()):
            winner.append(vote)
    return random.choice(winner)

def finding_nearest_neighbour(points,p,k=5):
    """Returns the indices of k nearest elements"""
    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i]=distance(points[i],p)
    ind=np.argsort(distances)
    return ind[0:k]
points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p=np.array([2.5,2])

plt.plot(points[:,0], points[:,1], "ro")
plt.plot(p[0], p[1], "bo")
plt.axis([0.5, 3.5, 0.5, 3.5])

def knn_predict(p, points, outcomes, k=5):
    #find k nearest neighbour
    ind=finding_nearest_neighbour(points,p,k)
    #predict the class of p based on majority vote
    return majority_vote(outcomes[ind])
#otcomes length sould be same as length of points
#each outcomes shows the class to which each point belongs
outcomes = np.array([0,0,0,0,1,1,1,1])

knn_predict(np.array([1,2.5]),points,outcomes,k=5)

def generate_synth_data(n=50):
    points=np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))),axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return(points, outcomes)

def make_prediction_grid(predictors,outcomes,limits,h,k):
    (x_min, x_max, y_min, y_max)=limits
    xs=np.arange(x_min,x_max,h)
    ys=np.arange(y_min,y_max,h)
    xx,yy=np.meshgrid(xs,ys)
    
    prediction_grid=np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p= np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)
    
(predictors, outcomes) = generate_synth_data()
k=5; filename="knn_synth.pdf"; limits=(-3,4,-3,4); h=0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits,h,k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

#iris dataset is a classical flower dataset which consist of 3 classes and four 
#categories of flower i.e length and width of sepal & petal of flower
from sklearn import datasets
iris= datasets.load_iris()
predictors=iris.data[:,0:2]
outcomes=iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],"ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1],"bo")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],"go")
k=5; filename="iris_1.pdf"; limits=(4,8,1.5,4.5); h=0.1
(xx,yy,prediction_grid)= make_prediction_grid(predictors, outcomes, limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid,filename)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors,outcomes)
sk_prediction=knn.predict(predictors)
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])
100*np.mean(sk_prediction==my_predictions)


