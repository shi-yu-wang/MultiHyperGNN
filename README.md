# MultiHyperGNN
Official implementation of MultiHyperGNN

Running environment:
Python 3.9
Pytorch 1.8.1
networkx 2.5
dgl 0.9.1
pandas 1.4.2
numpy, scipy

Dataset:
1. Genes:
	a. Input modes:
		Expression: expr_in_muscle.csv, expr_in_whole_blood.csv
		Graph: graph_in_muscle.csv, graph_in_whole_blood.csv
	b. Output modes:
		Expression: expr_out_lung.csv, expr_out_skin_not_sun_exposed.csv, expr_out_skin_sun_exposed.csv
		Graph: graph_out_lung.csv, graph_out_skin_not_sun_exposed.csv, graph_out_skin_sun_exposed.csv
2. Temperature:
	a. Early morning: 
		Value: t1.csv
		Graph: c1.csv
	b. Late morning:
		Value: t2.csv
		Graph: c2.csv
	c. Afternoon:
		Value: t3.csv
		Graph: c3.csv
	d. Night:
		Value: t4.csv
		Graph: c4.csv

Train the model:
1. To train the model, run: python trian.py.
	a. Note that train.py is written specifically for Genes, to train on Temperature, please replace with the corresponding Temperature data in both train.py and test.py.
	b. Trained model will be automatically saved as "model.pt".
	c. To change model parameters, please directly modify the parameters of model() in both train.py and test.py.
	d. To change training epochs, please change the value of "epochs" in train.py.
2. To test the model, run: python test.py.
