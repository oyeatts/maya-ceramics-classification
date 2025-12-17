from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import combinations
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.svm import SVC
from itertools import combinations
from mpl_toolkits.basemap import Basemap
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("ICPMS_RawData_LamoureuxStHilaire2017_translated.xlsx")
'''
DDIG - Spatial Organization Within A Traditional Political System: ICPMS Raw Data. Maxime Lamoureux-St-
Hilaire, Marcello Canuto. Middle American Research Institute, Tulane University, New Orleans. 2017 ( tDAR id: 428748) 
; doi:10.6067/XCV8FQ9ZQW 
'''
df['Subregion'] = "Peten Guatemala"
df = df.drop(index = 0)
bishop_df = pd.read_excel("Bishop_1995_pottery.xlsx")
'''
https://archaeometry.missouri.edu/murr_database.html
1995 	

INAA of Samples of Mayan Pottery for Ronald Bishop in 1995 at MURR

Unpublished research by Ronald Bishop
	Bishop, R.L.
'''
bishop_df = bishop_df.drop(columns = ["Material"])
lyle_df = pd.read_excel("Lyle_2000_pottery.xlsx")
'''
https://archaeometry.missouri.edu/murr_database.html
2000 	

Investigating Social Inequality: A Comparative Analysis of Late Classic Maya Residential Sites in the Three Rivers Region

Unpublished M.A. thesis, Department of Anthropology, University of Texas at San Antonio
	Lyle, A.S. 	Pottery
'''
lyle_df = lyle_df.drop(columns = "Subregion")
lyle_df["Investigator"] = 'Lyle, A.S.'
df4 = pd.read_excel("FultonKA_Diss-Raw-Data_tDAR_translated.xls")
'''
Raw Artifact & Chemical Data - Community Identity and Social Practice during the Terminal Classic Period at
Actuncan, Belize. Kara Fulton. 2015 ( tDAR id: 398973) ; doi:10.6067/XCV83R0V68 
'''
df4 = df4.drop(columns = "Row Id")
df4 = df4.drop(index = 0)
df4['Subregion'] = "Western Belize"

#df5 = pd.read_excel("MesoRAD-v.1.4.shiftedloc_translated.xlsx")
def getstatistic(s):
    return stats.shapiro(s).statistic


m = Basemap(projection='merc',llcrnrlat=14.5,urcrnrlat=22,\
            llcrnrlon=-93,urcrnrlon=-86,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua') 
plt.title("Maya Area")
plt.show()

df_train = pd.merge(lyle_df, bishop_df, how='outer')
df_train = df_train.drop(columns = 'ANID')
yt = df_train['Investigator']
yt_map = LabelEncoder().fit_transform(yt)
X_df = df_train.drop(columns = 'Investigator')

df_train = df_train.drop(columns = 'Investigator')
df_train['Investigator'] = yt_map

lyle_locs = df_train['Investigator'] == 1
bishop_locs = df_train['Investigator'] == 0

lyle_locs = np.asarray(lyle_locs)
bishop_locs = np.asarray(bishop_locs)

lyle_X = df_train.iloc[lyle_locs]
bishop_X = df_train.iloc[bishop_locs]

labels = list(bishop_X.columns)

bishop_test = {}
lyle_test = {}

#testing whether element has normal distribution within respective sample
for i in labels:
    if i == 'Investigator':
        break
    col_bishop = bishop_X[i]
    col_lyle = lyle_X[i]
    
    shapiro_bishop = stats.shapiro(col_bishop)
    shapiro_lyle = stats.shapiro(col_lyle)
    
    bishop_test.setdefault(i, list(shapiro_bishop))
    lyle_test.setdefault(i, list(shapiro_lyle))
    
    mct_bishop = stats.monte_carlo_test(col_bishop, stats.norm.rvs, getstatistic, alternative='less')
    mct_lyle = stats.monte_carlo_test(col_lyle, stats.norm.rvs, getstatistic, alternative='less') 
    extremities_bishop = np.where(np.linspace(0.8, 1.0, 100) <= mct_bishop.statistic)[0]
    if(shapiro_bishop[1] > 0.05):
        print("Only showing tests in which the p value was above 0.05, otherwise it is statistically insignificant. \n")
        fig, ax = plt.subplots()
        
        ax.hist(mct_bishop.null_distribution, density=True, bins = np.linspace(0.8, 1.0, 100))
        ax.set_title('Shapiro-Wilk Test Distributions for Bishop, element = ' + i)
        ax.set_xlabel('statistic')
        ax.set_ylabel('probability density')
        #plt.legend('p-value (red) = ' + str(shapiro_bishop[1]))
        
        #extremities_bishop = np.where(np.linspace(0.8, 1.0, 100) <= mct_bishop.statistic)[0]
        
        for j in extremities_bishop:
            ax.patches[j].set_color('r')
            
        plt.xlim(0.90, 1.0)
        plt.ylim(0, 0.4)
        plt.show()
    if(shapiro_lyle[1] > 0.05):
        print("Only showing tests in which the p value was above 0.05, otherwise it is statistically insignificant. \n")
        fig2, ax2 = plt.subplots()
        ax2.hist(mct_lyle.null_distribution, density=True, bins = np.linspace(0.80, 1.0, 100))
        ax2.set_title('Shapiro-Wilk Test Distributions for Lyle, element = ' + i)
        ax2.set_xlabel('statistic')
        ax2.set_ylabel('probability density')
        #plt.legend('p-value (red) = ' + str(shapiro_lyle[1]))
    
        extremities_lyle = np.where(np.linspace(0.80, 1.0, 100) <= mct_lyle.statistic)[0]
        for p in extremities_bishop:
            ax2.patches[p].set_color('r')
        plt.xlim(0.85, 1.0)
        plt.ylim(0, 5)
        plt.show()
        ''' Learn about pipelinging to make testing more effecient'''

pca = PCA(n_components=2).fit_transform(X_df)
pca = pd.DataFrame(pca)

model = KMeans(n_clusters = 3)
model.fit(X_df)
clusters = model.labels_
centroids = model.cluster_centers_
sse = model.inertia_
sse_dict = {}
ratio = {}

dist = linkage(X_df, method ='average', metric = 'euclidean')

plt.figure(figsize=(10, 10))
dendrogram(dist, color_threshold=4)
y = pca[1]
ax = plt.gca()
labels = [y[int(t.get_text())] for t in ax.get_xticklabels()]
ax.set_xticklabels(labels)
plt.xticks(rotation = 90)
plt.show()
#Best: [V, Ca], [Th,Zn], [Sc,V], [Cr,v], [U,Th], [U,Hf](for all knn)
#new best: ['Zn', 'Th', 'U'], ['Sc', 'Th', 'U'], ['Ca', 'Zn', 'Th'],
#['Ca', 'Zn', 'V', 'Th', 'U'], 	['Ca', 'Sc', 'Th', 'U', 'Hf'],
#2(3)-['Sc', 'Cr', 'Th']{20}, 1-['Cr', 'Th', 'U']{26}, 2(3)-['Ca', 'Th', 'U']{20}

#X = df_train[['Ca','Zn','Sc','V','Cr','Th','U', 'Hf']]
#takes most frequent classes from early tests and tests all combs of 3
#X = df_train[['Sc', 'Zn', 'Ca', 'Cr', 'V', 'Th', 'U']]

#X = [['Cr', 'Th', 'U'], ['Sc', 'Cr', 'Th'], ['Ca', 'Th', 'U']]

#found that ['Ca', 'Th', 'U'] is almost perfect after testing above
#~99 accuracy, p, and r with manhattan metric

#['U', 'V', 'Th', 'Ni'], and ['U', 'Ca', 'Th', 'Ni'] with NCA and manhattan work well

X = df_train[['As', 'U', 'Eu', 'Sc', 'Th', 'Al', 'Ti']]
#results_KNN_best = pd.DataFrame()
#r = 24
#for r in range(24,1, -1):

    #X = df_train[['As','U','V', 'Cr', 'Ca', 'Th', 'Al', 'Eu', 'Fe', 'Ni']]
    #X = list(X_df.columns)
    #X = df_train[X_new]
    
pca = make_pipeline(StandardScaler(), PCA(n_components=2))
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2))
    
reductions = [("PCA", pca), ("NCA", nca)]
metrics = ['manhattan']
    
accs = []
pres = [] 
recs = []
knns = []
class_labels = []
algorithms = []
X_vars = []
reduction_methods = []
metrics_list = []
scores = []
accs_mean = []
pres_mean = []
recs_mean = []
scores_mean = []
predictions_knn = []
'''
pca0 = np.asarray(pca[0])
pca1 = np.asarray(pca[1])
pca0 = np.reshape(pca0, (-1,1))
pca1 = np.reshape(pca1, (-1,1))
'''
for p in range(7,8): # replaced with r when looping to find best params
    combs = list(combinations(X, p))
    #combs = X
    for i in combs:
        i = list(i)
        Xt = df_train[i]
    
        Xt = np.asarray(Xt)
        for n in range(2, 7):
            for met in metrics:
                '''
                    model = KMeans(n_clusters = n, algorithm='lloyd')
                    acc = accuracy_score(yt_map, yp)
                    rec = recall_score(yt_map, yp, average = 'macro')
                    pre = precision_score(yt_map, yp, average = 'macro')
                    '''
                    
                knn =  KNeighborsClassifier(n_neighbors=n, n_jobs=-1, algorithm = 'auto', metric = met)
                    
                for m, (name, model) in enumerate(reductions):
                    for v in range(100): #run for avg n times
                        X_train,X_test, y_train, y_test = train_test_split(Xt, yt_map, test_size=0.3, stratify=yt_map) 
                        model.fit(X_train, y_train)
                            
                        knn.fit(model.transform(X_train), y_train)
                            
                        yp = knn.predict(model.transform(X_test))
                            
                        acc_knn = knn.score(model.transform(X_test), y_test)
                            
                        X_embed = model.transform(Xt)
                            
                        accs_mean.append(accuracy_score(y_test, yp))
                        pres_mean.append(precision_score(y_test, yp, average='macro'))
                        recs_mean.append(recall_score(y_test, yp, average = 'macro'))
                        scores_mean.append(acc_knn)
        
                    class_labels.append(knn.classes_)
                    algorithms.append(knn.algorithm)
                    metrics_list.append(knn.effective_metric_)
                    knns.append(n)
                    X_vars.append(i)
                    score_mean = np.mean(scores_mean)
                    scores.append(score_mean)
                    reduction_methods.append(name)
                    accs.append(np.mean(accs_mean))
                    pres.append(np.mean(pres_mean))
                    recs.append(np.mean(recs_mean))
                      
                    scores_mean.clear()
                    pres_mean.clear()
                    recs_mean.clear()
                    accs_mean.clear()
                     
                    #HEEEEEEEREE+++++++++ make a confusion matrix
                       
                    plt.figure()
                    DecisionBoundaryDisplay.from_estimator(knn, X_embed, response_method='predict')
                    plt.scatter(X_embed[:,0], X_embed[:,1], c=yt_map, cmap='rainbow')
                    plt.title("KNN = " + str(n) + ": Test Mean Accuracy = " + str(score_mean))
                    plt.xlabel(name + "[" + i[0] + ", " + i[1] + ", " + i[2] + "]" + "[0]")
                    plt.ylabel(name + "[" + i[0] + ", " + i[1] + ", " + i[2] + "]" + "[1]")
                       
                    cm = ConfusionMatrixDisplay.from_estimator(knn, model.transform(X_test), y_test)
                       
                    plt.show()
              
                
results_KNN = pd.DataFrame()
results_KNN["Train_vars"] = X_vars
results_KNN['KNN'] = knns
results_KNN['Mean_Acc'] = scores
results_KNN["Reduction"] = reduction_methods
results_KNN["Accuracy"] = accs
results_KNN["Precision"] = pres
results_KNN["Recall"] = recs
results_KNN["Class_labels"] = class_labels
results_KNN["Algorithm"] = algorithms
results_KNN["Metric"] = metrics_list

    #best_index = scores.index(np.max(scores))
    
    #best = results_KNN.iloc[best_index]
    
    #results_KNN_best[r] = best
    #X_new = results_KNN_best.at["Train_vars", r]

#results_KNN_best = results_KNN_best.T

#results_KNN_best.to_csv("KNN_Best.csv", index_label="# of Attributes")
'''        
print("acc score =" + str(acc))
print("rec score =" + str(rec))
print("pre score =" + str(pre))

accs.append(acc)
recs.append(rec)
pres.append(pre)

model.fit(Xt, yt_map)

clusters = model.labels_
centroids = model.cluster_centers_
sse = model.inertia_
print(sse)
            
print(n)
ratio[round(sse/n, ndigits= 4)] = n
sse_dict[n] = sse
if n in range(1, 81): 
    plt.figure()
    plt.scatter(pca[0], y, c = clusters, cmap='tab10')
    plt.plot(centroids[:,0], centroids[:,1], 'k+', markersize = 12)
    plt.title('SSE = %.3f' % sse)
    plt.show()
           
plt.figure()
plt.scatter(pca[0], pca[1])
DecisionBoundaryDisplay.from_estimator(, pca[0], response_method='predict')
plt.show()

print("acc max = " + str(max(accs)))
print('rec max = ' + str(max(recs)))
print('pre max = ' + str(max(pres)))
'''
print('knn max = ' + str(max(scores)))

#DECISION TREE

def print_tree(model, criterion, ccp_alpha, atts):
#for tree in results['estimator']:
    plt.figure(figsize=(10,10))
    
    plot_tree(model, filled = True, proportion=True)
    plt.title(("atts: " + str(atts) + "criterion: " + str(criterion) + " | ccp_alpha: " + str(ccp_alpha)), fontsize = 18)
    plt.show()

#X_columns = df_train[['As', 'U', 'Eu', 'Sc', 'Th', 'Al', 'Ti']]

#X_columns = df_train[['As', 'U', 'K'], ['U', 'Ni', 'K']] #- works well ~98%
#X_columns = df_train[['U', 'K']] #- ~98%
X_columns = df_train[['As', 'U', 'K', 'Ni']]
#X_columns = df_train[['As', 'La', 'Lu', 'Nd', 'Sm', 'U', 'Yb', 'Ce', 'Co', 'Cr', 'Cs', 'Fe', 'Ta']]
#X_columns = list(X_df.columns)
#X_columns = df_train[['Ca','Th','U']]

#pca = make_pipeline(StandardScaler(), PCA(n_components=2))
#nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2))

#reductions = [("PCA", pca), ("NCA", nca)]

accs_DTC = []
pres_DTC = []
recs_DTC = []
class_labels_DTC = []
X_vars_DTC = []
#scores_DTC = []
accs_mean_DTC = []
pres_mean_DTC = []
recs_mean_DTC = []
#scores_mean_DTC = []
predictions_DTC = []
criterion_DTC = []
ccp_alpha_DTC = []

correlation = np.corrcoef(X_df, yt_map, rowvar=False)
correlation_df = pd.DataFrame(correlation, index = df_train.columns, columns = df_train.columns)

for p in range(3,4):
    combs = list(combinations(X_columns, p))
    #combs = X
    for i in combs:
        i = list(i)
        X = df_train[i]
        X = np.asarray(X)
        for c in ['entropy']: #tried gini was slightly worse than entropy
            for j in range (5,11):
                j_sub = j / 100
                #plt.figure()
                dtc = DecisionTreeClassifier(criterion = c, ccp_alpha= j_sub)
                criterion_DTC.append(c)
                ccp_alpha_DTC.append(j_sub)
                dtc.fit(X, yt_map)
                yp_DTC = cross_val_predict(dtc, X, yt_map, cv = 5)
                
                print_tree(dtc, c, j_sub, i)
                
                cm = ConfusionMatrixDisplay.from_predictions(yt_map, yp_DTC)
                plt.show()
                
                class_labels_DTC = dtc.classes_
                X_vars_DTC.append(i)
                accs_DTC.append(accuracy_score(yt_map, yp_DTC))
                pres_DTC.append(precision_score(yt_map, yp_DTC, average='macro'))
                recs_DTC.append(recall_score(yt_map, yp_DTC, average='macro'))
                #print("criterion = " + i + ", ccp_alpha = " + str(j_sub))
                #print('acc = %.3f, pre = %.3f, rec = %.3f' % (acc, pre, rec))
                #plot_tree(model, filled=True)
                #plt.show()

results_DTC = pd.DataFrame()
results_DTC['Train_Vars'] = X_vars_DTC
#results_DTC["Class_labels"] = class_labels_DTC
results_DTC["Accuracy"] = accs_DTC
results_DTC["Precision"] = pres_DTC
results_DTC["Recall"] = recs_DTC
results_DTC['Criterion'] = criterion_DTC
results_DTC['CCP_Alpha'] = ccp_alpha_DTC



#SVM
#set class_weight param to balanced?

#X_columns = df_train[['Sc', 'Zn', 'Ca', 'Cr', 'V', 'Th', 'U']]
X_columns = df_train[['Ca','Th','U']]

pca = make_pipeline(StandardScaler(), PCA(n_components=2))
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2))

reductions = [("PCA", pca), ("NCA", nca)]

SVC_c = [1.0]
SVC_kernel = ['linear', 'poly'] #sigmoid and rbf don't work well
SVC_degree = [2,3]
SVC_gamma = ['scale']
SVC_coef = [0.0]
SVC_tol = [0.01, 0.001, 0.1]

SVC_c_list = []
SVC_kernel_list = []
SVC_degree_list = []
SVC_gamma_list = []
SVC_coef_list = []
SVC_tol_list = []

accs_SVC = []
pres_SVC = []
recs_SVC = []
scores_SVC = []
SVC_class_labels = []
for p in range(3,4):
    combs = list(combinations(X_columns, p))
    #combs = X
    for i in combs:
        i = list(i)
        X = df_train[i]
        X = np.asarray(X)
        for c in SVC_c:
            for j in SVC_kernel:
                for d in SVC_degree:
                    for g in SVC_gamma:
                        for co in SVC_coef:
                            for t in SVC_tol:
                                
                                svc = SVC(cache_size=1000, C=c, kernel=j, degree=d,
                                          gamma= g, coef0=co, tol=t)
                                
                                for m, (name, model) in enumerate(reductions):
                            
                                    model.fit(X, yt_map)
                                        
                                    #svc.fit(model.transform(X), yt_map)
                                        
                                    yp_SVC = cross_val_predict(svc, model.transform(X), yt_map)
                                        
                                    #acc_svc = svc.score(model.transform(X), y_test)
                                        
                                    X_embed = model.transform(X)
                                    acc = accuracy_score(yt_map, yp_SVC)
                                    
                                    SVC_class_labels.append(i)
                                    SVC_c_list.append(c)
                                    SVC_kernel_list.append(j)
                                    SVC_degree_list.append(d)
                                    SVC_gamma_list.append(g)
                                    SVC_coef_list.append(co)
                                    SVC_tol_list.append(t)
                                    accs_SVC.append(acc)
                                    pres_SVC.append(precision_score(yt_map, yp_SVC, average='macro'))
                                    recs_SVC.append(recall_score(yt_map, yp_SVC, average = 'macro'))
                                    #scores_SVC.append(acc_svc)
                                    
                                    plt.figure()
                                    svc.fit(model.transform(X), yt_map)
                                    DecisionBoundaryDisplay.from_estimator(svc, X_embed, response_method='predict')
                                    plt.scatter(X_embed[:,0], X_embed[:,1], c=yt_map, cmap='rainbow')
                                    plt.title('SVC - degree = %d, kernel = %s, c = %.2f, tol = %.3f, acc = %.4f' % (d, j, c, t, acc ))
                                    plt.xlabel(name + "[" + i[0] + ", " + i[1] + ", " + i[2] + "]" + "[0]")
                                    plt.ylabel(name + "[" + i[0] + ", " + i[1] + ", " + i[2] + "]" + "[1]")
                                   
                                    cm = ConfusionMatrixDisplay.from_predictions(yt_map, yp_SVC)
                                    
                                    plt.show()
                                    
results_SVC = pd.DataFrame()
#results_DTC['Train_Vars'] = X_vars_DTC
results_SVC["Class_labels"] = SVC_class_labels
results_SVC["Accuracy"] = accs_SVC
results_SVC["Precision"] = pres_SVC
results_SVC["Recall"] = recs_SVC
results_SVC['C'] = SVC_c_list
results_SVC['Kernel'] = SVC_kernel_list
results_SVC['Degree'] = SVC_degree_list
results_SVC['Gamma'] = SVC_gamma_list
results_SVC['Coef'] = SVC_coef_list
results_SVC['Tolerance'] = SVC_tol_list
