import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import IsolationForest
import collections
from sklearn.decomposition import PCA


# Load training data
data = pd.read_csv('train_data.txt', sep='\t')
smiles = data.iloc[:,1]


# Calculate RDKit descriptors
descr = Descriptors._descList[0:2] + Descriptors._descList[3:]
calc = [x[1] for x in descr]

def describe(mols):
	descrs = []
	for mol in mols:
		fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=2048)		
		fp_list = []
		fp_list.extend(fp.ToBitString())
		fp_expl = [float(x) for x in fp_list]
		ds_n = []
		for d in calc:
			v = d(mol)
			if v > np.finfo(np.float32).max:
				ds_n.append(np.finfo(np.float32).max)
			else:
				ds_n.append(np.float32(v))
		
		descrs += [fp_expl + list(ds_n)];
	
	return descrs


mols = [Chem.MolFromSmiles(s) for s in smiles]
x = np.array(describe(mols))


test = pd.read_csv('nonagg_data_test.txt', sep='\t')
smiles_test = test.iloc[:,2]
mols_test = [Chem.MolFromSmiles(s) for s in smiles_test]
x_test = np.array(describe(mols_test))


clf = IsolationForest(n_estimators=1000, random_state=1, contamination=0.29).fit(x)
preds = clf.predict(x_test)
collections.Counter(preds)


def classano(x):
	if x == 'AGG':
		return '-1'
	else:
		return '1'



seeds = [*range(0, 20, 1)]

'''
pca = PCA(n_components=10)
x2 = pca.fit_transform(x)
'''


iF_retrieval = []

for i in seeds:
	iF = IsolationForest(n_estimators=1000, random_state=i, contamination=0.29).fit(x)
	preds_train = iF.predict(x)
	df_preds = pd.DataFrame(preds_train, columns = ['Preds'], index=None)
	data_all = pd.concat([data, df_preds], axis=1)
	data_all['transform'] = [classano(x) for x in data_all['agg']]
	data_all['transform'] = data_all['transform'].astype(str).astype(int)
	data_all2 = data_all[data_all['agg'] == 'AGG']
	data_all2['results'] = data_all2['Preds'] == data_all2['transform']
		frac_iF_correct = [collections.Counter(data_all2['results'])[1] / len(data_all2)]
	iF_retrieval += frac_iF_correct


np.mean(iF_retrieval)*100
np.std(iF_retrieval)*100




'''
Full descriptor set:
Repeated n=20
average retrieval from training data --> 27%
std --> 1%


PCA (10 components):
Repeated n=20
average retrieval from training data --> 29.18250950570342%
std --> 0.6894092914246679%

'''

