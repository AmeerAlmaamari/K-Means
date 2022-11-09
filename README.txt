To run the attached codes you need the following packages:

numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
seaborn==0.11.1
tqdm==4.59.0
---------------------------------

To open the datasets you need to change the path in the code, and import them with pandas

animals = pd.read_csv(r'...\animals',delimiter=' ',header=None)
countries = pd.read_csv(r'...\countries',delimiter=' ',header=None)
fruits = pd.read_csv(r'...\fruits',delimiter=' ',header=None)
veggies = pd.read_csv(r'...\veggies',delimiter=' ',header=None)

--------------------------------------------------------------
Also, to active l2 norm you should just convert l2_norm from False to True
--------------------------------------------------------------------------
To choose either Manhatten or Eculdien distances. Change the name of "method" parameter in the called object of the class.
