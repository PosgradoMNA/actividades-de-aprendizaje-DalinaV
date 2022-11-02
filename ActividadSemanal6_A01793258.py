#!/usr/bin/env python
# coding: utf-8

# #**Maestría en Inteligencia Artificial Aplicada**
# ## *Ciencia y Analítica de Datos*
# 
# 
# ####**Curso: Ciencia y analítica de datos (Gpo 10)**
# 
# 
# 
# ####**Actividad Semanal -- 6, visualización**
# 
# 
# ####**Prof.Maria de la Paz Rico**
# 
# 
# ####**01 de noviembre de 2022**
# 
# 
# Nombre del estudiante: 
# ***Dalina Aidee Villa Ocelotl (A01793258)***
# 
# 
# 
# 

# **Importar librerías**

# In[254]:


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# **Lectura de la data a utilizar**

# Del ejercicio **Limpieza de una base de Datos**

# In[255]:


path='https://raw.githubusercontent.com/PosgradoMNA/Actividades_Aprendizaje-/main/default%20of%20credit%20card%20clients.csv'
df = pd.read_csv(path).iloc[:, 1:] ### Seleccionamos todo menos la primer columna, al ser el índice de la escritura anterior


# 
# 
# ---
# 
# 
# # **Paso 2:** Obten la información del DataFrame con los métodos y propiedades: shape, columns, head(), dtypes, info(), isna()**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[256]:


df.head(3)


# In[257]:


df.shape


# In[258]:


df.columns # nombres de las columnas


# In[259]:


df.info() 
# Informacion general de los datos de cada cloumna
# Indica el numero de filas del dataset
# Muestra el numero de datos No Nulos por columna (valores validos)
# Tipo de dato de cada columna
# Tamaño total del dataset


# In[260]:


df.dtypes #observamos los tipos de datos que se tienen


# In[261]:


df.describe()


# In[262]:


# verificar cuales valores son NaN o nulos (Null)
df.isna()


# 
# ---
# 
# # **Paso 3:** Limpia los datos eliminando los registros nulos o rellena con la media de la columna**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[263]:


df.isnull().values.any()


# In[264]:


df.isna().any()


# In[265]:


df1=df[~np.isnan(df).any(axis=1)]
df1


# In[266]:


df1.isna().any()


# 
# ---
# 
# # **Paso 4:** Calcula la estadística descriptiva con describe() y explica las medidas de tendencia central y dispersión**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[267]:


df1.describe()


# Se describe como se comportan las variables en su distribucion. Incluye las medidas de tendencia central en percentiles para ver hasta que valor acumula mas variaza respecto de la media y con ello se observa si es necesario resalizar algun tipo de estandarizacion de los datos.
# 
# En la primer variable como el monto del crédito se observa que se han otorgado hasta 1 millon de pesos y como minimo de 10 mil por lo que se puede observar que estan dispersos los datos.

# In[ ]:





# ******Descripcion de las variables: Attribute Information:
# 
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# 
# X1: Amount of the given credit *(NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. X2: 
# 
# *Gender (1 = male; 2 = female). X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
# 
# X4: ** MaritalStatus** (1 = married; 2 = single; 3 = others). X5: Age (year). X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

# In[268]:


df1 = df1.rename({"X1":"CreditAmount",  "X2":"Gender", "X3":"Education", "X4":"MaritalStatus","X5":"Age"}, axis='columns')


# In[269]:


df1.head()


# 
# ---
# 
# # **Paso 5:** Realiza el conteo de las variables categóricas**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[270]:


CreditAmount=df1[df1['CreditAmount']<=0]
print(CreditAmount) 


# In[271]:


df1['CreditAmount'].plot.hist()  # comprobamos si hay montos negativos en los creditos, pues no puede haver montos negativos de prestamos


# In[272]:


df1[df1['CreditAmount']<=0]=0  # si los hubiera aqui los cambiariamos a cero de acuerdo a algunos menotodos de estadarizacion


# In[273]:


df1['Gender'].plot.hist()  #verificamos que todos los datos son valores 1 o 2 respectivamente por genero


# In[274]:


Gender=df1[(df1['Gender']!=1) & (df1['Gender']!=2)]
print(Gender)


# In[213]:


#df1[(df1['Gender']!=1) & (df1['Gender']!=2)]=np.random.randint(1,3) # solo se encontro un dato por lo que se puede insertar de manera aletoria


# In[275]:


df1['Gender'].plot.hist() 


# In[215]:


df1['Education'].plot.hist()  #verificamos que todos los datos son valores (1 = graduate school; 2 = university; 3 = high school; 4 = others)


# Se observa que para el nivel de estudios se tienen otras categorias distintas. Por lo que se considera conveniente categorizar como "otros_estudios"

# In[216]:


Education=df1[(df1['Education']!=1) & 
                           (df1['Education']!=2) & 
                           (df1['Education']!=3) &
                           (df1['Education']!=4)]
print(Education)


# In[217]:


df1[(df1['Education']!=1) & 
                 (df1['Education']!=2) & 
                 (df1['Education']!=3) & 
                 (df1['Education']!=4)]=4 #se crea la nueva categoria


# In[218]:


df1['MaritalStatus'].plot.hist()  # se valida que cumpla con ls supuestos (1 = married; 2 = single; 3 = others)


# Se observa que en la variable de estado civil existen otras categorias

# In[219]:


MaritalStatus=df1[(df1['MaritalStatus']!=1) & 
                               (df1['MaritalStatus']!=2) & 
                               (df1['MaritalStatus']!=3)]
print(MaritalStatus)


# In[220]:


df1[(df1['MaritalStatus']!=1) &
                 (df1['MaritalStatus']!=2) &
                 (df1['MaritalStatus']!=3)]=3 # se trata de unificar la variable de otros estados civiles


# In[221]:


Age=df1[df1['Age']<18]
print(Age)
df1[df1['Age']<18]=18 # Ajustamos el valor de la edad al minimo aceptable que es 18 años


# In[222]:


df1['Age'].plot.hist() # se valida la distribucion de valores de la variable Edad


# In[223]:


df1.head()


# In[224]:


#Para conocer el estatus de comportamiento de pago de los ultimos 6 meses para ver si es pago sostenido
r6=df1.columns.get_loc('X6') #Se busca el índice para la columna llamada 'X6'
r11=df1.columns.get_loc('X11') #Se busca el índice para la columna llamada 'X11'
for i in range(r6,r11+1):
  x6_x11=df1[df1.iloc[:,i]<-1] 
  print(x6_x11)
  x6_x11.plot.box()
#Se observa que hay valores negativos de todo tipo en esas columnas de pago


# 
# ---
# 
# # **Paso 6:** Escala los datos, si consideras necesario**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[225]:


df1 = df1.rename({"X1":"CreditAmount",  "X2":"Gender", "X3":"Education", "X4":"MaritalStatus","X5":"Age", "X6":'Status_Sep', "X7":'Status_Ago', "X8":'Status_Jul', "X9":'Status_Jun', "X10":'Status_May', "X11":'Status_Abr'}, axis='columns')


# In[226]:


## Sólo variables numéricas
n_df = df1.drop(columns=['Y','Gender','Education','MaritalStatus', 'Status_Sep', 'Status_Ago', 'Status_Jul', 'Status_Jun', 'Status_May', 'Status_Abr'])


# Al ser un análisis númerico, cuya intención es capturar la mayor cantidad de información (proporción de varianza) con las menores variables posibles, se decidió quitar las siguientes variables: 
# 
# * 'ID' - Identificador del individuo
#   
# * 'Y' (flag del comportamiento a predecir)
#   
# * Variables categóricas ('Gender','Education','MaritalStatus', 'Status_Sep', 'Status_Ago', 'Status_Jul', 'Status_Jun', 'Status_May', 'Status_Abr')

# In[227]:


n_df.head()


# In[228]:


n_df.shape


# In[229]:


from sklearn import preprocessing


# In[230]:


sc_n_df = preprocessing.scale(n_df.dropna(axis=0)) ## Estandarizamos las variables para tenerlos en las mismas unidades, y centramos en la media
sc_n_df = pd.DataFrame(sc_n_df, columns = n_df.columns)


# In[231]:


columnas_n_df = sc_n_df.columns # esta lista de columans la utilizaremos más adelante para mostrar los resultados


# In[232]:


sc_n_df.head()


# 
# ---
# 
# # **Paso 7:** Reduce las dimensiones con PCA, si consideras necesario.
# 7.1 .- Indica la varianza de los datos explicada por cada componente seleccionado. Para actividades de exploración de los datos la varianza > 70%
# 
# 
# 
# 7.2 .-Indica la importancia de las variables en cada componente**
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[233]:


pcanalisis = PCA(n_components=13) ## Son 13 componentes al ser sólo 14 variables, el máximo número de componentes principales posibles que reduzcan la dimensionalidad sería 13.


# Se decidió calcular todos los componentes principales para después realizar el análisis y sólo quedarnos con los adecuados y ver en donde se podria englobar la mayor cantidad de varianza 

# In[234]:


pcanalisis.fit(sc_n_df) ## Ajuste del PCA con información numérica disponible


# **Resumen de resultados**

# In[235]:


pca_resumen = pd.DataFrame({'Desviación_Estándar': np.sqrt(pcanalisis.explained_variance_), 
                          'Prop_Varianza_explicada': pcanalisis.explained_variance_ratio_,
                          'Acum_Prop_Varianza_explicada': np.cumsum(pcanalisis.explained_variance_ratio_)
                          })
pca_resumen = pca_resumen.transpose()
pca_resumen.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']
pca_resumen.round(2)


# Observamos gráficamente la proporción de información capturada en el PCA (utilizando Scree Plot)

# In[236]:


## Se utilizó las siguientes líneas de código aprendidas y replicadas en clase

PC_components = np.arange(pcanalisis.n_components_) + 1
#PC_components

_ = sns.set(style = 'whitegrid', 
            font_scale = 1.2
            )

fig, ax = plt.subplots(figsize=(10, 7))

_ = sns.barplot(x = PC_components, 
                y = pcanalisis.explained_variance_ratio_, 
                color = 'b',
                label='Varianza explicada',
                )

_ = sns.lineplot(x = PC_components-1, 
                 y = np.cumsum(pcanalisis.explained_variance_ratio_), 
                 color = 'black', 
                 linestyle = '-', 
                 linewidth = 2, 
                 marker = 'o', 
                 markersize = 8, 
                 label = 'Varianza Acumulada'
                 )

plt.title('Scree Plot')
plt.xlabel('N-th Principal Component')
plt.ylabel('Variance Explained')
plt.legend(loc='lower right')
plt.ylim(0, 1)

plt.show()


# 
# 
# Con los resultados anteriores, de la tabla (PCA_resumen) y la gráfica (Scree Plot), concluimos que es pertinente utilizar los primeros 7 Componentes Principales, ya que con esto capturamos el 87% de la varianza del total de variables numéricas. Aunque, observamosque en los primeros 2 componentes se agrupa la mayor cantidad de varianza pero es hasta el 13 avo componente donde se agrupa toda la varianza.
# 
# En otras palabras, transoformando las variables a estos compomentes (PC1, PC2, PC3, PC4, PC5, PC6, PC7) y sólo utilizando esta información, reducimos de 14 a 7 variables (50% de las numéricas), capturando el 87% de la información. 
# 
# No utilizamos más componentes porque cada componente extra aporta a más 5% extra de información, aumentando la dimensionalidad por lo que no es un lift significativo en la información a utilizar, recordemos que al final es una transformacion de los datos originales.

# In[237]:


pcaComponentes_df = pd.DataFrame(pcanalisis.components_.round(4).transpose(), 
                                columns=['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'],
                                index= columnas_n_df
                                )
pcaComponentes_df


# **PC 1**

# In[238]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 1:')
print('\n')
print(pcaComponentes_df.PC1.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 1:')
print('\n')
print(pcaComponentes_df.PC1.nsmallest())


# **INTERPRETACIÓN**
# 
# Observamos que de la retroalimenacion de los componenetes podemos ver que es hasta 0.3 donde se guarda la mayoria de la varianza, por lo que no ha un componente unico donde se agrupe mucha variaznza. Esto tambien asociado al tipo de vsriables que se estan analizando.

# **PC 2**

# In[239]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 2:')
print('\n')
print(pcaComponentes_df.PC2.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 2:')
print('\n')
print(pcaComponentes_df.PC2.nsmallest())


# **INTERPRETACIÓN**
# 
# Observamos que agruoa mayor varianza que el primer componente aunque sabemos que en donde se agrupa la mayor vvarianza derivado de la transformacion ortogonal son los primeros dos comnentes

# **PC 3**

# In[240]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 3:')
print('\n')
print(pcaComponentes_df.PC3.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 3:')
print('\n')
print(pcaComponentes_df.PC3.nsmallest())


# **INTERPRETACIÓN**
# 
# Se observa que en el terce componente principal a pesar de se de variables de pagos terminan obteniendo las mas relevantes para otorgar o no un prestamo. Sin embargo los pagos de los meses mas recientes captura menos varianza que algunos mas anteriores, loq ue valida la hipotesis de que es mejor tener mayor tiempo de historial dentro de la ventana de los 6 meses

# **PC 4**

# In[241]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 4:')
print('\n')
print(pcaComponentes_df.PC4.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 4:')
print('\n')
print(pcaComponentes_df.PC4.nsmallest())


# **INTERPRETACIÓN**
# 
# Se observa que en el cuarto componente al igual que en los primeros dos, las principalees variables son saldos de los meses, sin embargo mucho tiene que ver con la consistencia de cada variable ya ue algunos pagos mensuales muestran poca relevancia. En este componente las variables ninguna muestra mas del 30% de la explicabilidad.

# **PC 5**

# In[242]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 5:')
print('\n')
print(pcaComponentes_df.PC5.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 5:')
print('\n')
print(pcaComponentes_df.PC5.nsmallest())


# **INTERPRETACIÓN**
# 
# 
# Como se sabe atraves de componenetes principales la transofrmacion de la variable trata de garantizar cumplir los supuestos de la funcion de optimizacion, por lo que despues de algunos componentes el resultado es la transformacion de una transformacion, donde vemos que a diferencia del componente numero 4 ahora las variables de pagos agrupan mayor parte de la varianza.

# **PC 6**

# In[243]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 6:')
print('\n')
print(pcaComponentes_df.PC6.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 6:')
print('\n')
print(pcaComponentes_df.PC6.nsmallest())


# **INTERPRETACIÓN**
# 
# 
# Observanos que la importancia de las variables va cambiando de acuerdo a la aplicacin de cada transformacion pues cada refinanmiento agrupa mayor correlacion de los datos, se presenta la variable de edad con poca importancia y la variable de pagos describiendo la mayor importancia de este componenete, por lo que se recomendaria seria agrupar una variable resumen del comportamiento de pago mensual de los clientes.

# **PC 7**

# In[244]:


print('Variables MÁS importantes, y su coeficiente para el componente Principal 7:')
print('\n')
print(pcaComponentes_df.PC7.nlargest())
print('\n')
print('Variables MENOS importantes, y su coeficiente para el componente Principal 7:')
print('\n')
print(pcaComponentes_df.PC7.nsmallest())


# **INTERPRETACIÓN**
# 
# 
#  De acuerdo a los componentes anteriores, este el el unico que considera el monto del credito como una variable imporntae aunque con muy baja explicacion, esto nos muestra que a mayor refinanmiento del numero de componentes principales gran parte de lo que se establece en los primeros componentes es mas adecuado. Ademas como la variable de saldo es parte relevante de las los componentes principales, tambien podria resumirse de manera trimestral el comportamiento de saldo.
# 
# 

# Para las variables numericas

# In[245]:


n_df.columns


# In[246]:


n_df.describe()


# In[247]:


columnas_n_df[0]


# In[248]:


sns.set(rc={'figure.figsize':(20,40)})
fig, axes = plt.subplots(7,2 )    
  
for k in range(0,14):
  plt.subplot(7,2,k+1) 
  sns.boxplot(data=n_df, x=columnas_n_df[k], saturation=1)
  plt.xlabel(columnas_n_df[k])  

plt.show()


# **INTERPRETACIÓN DEL DESCRIBE Y DEL BOXPLOT**
# 
# 
# Se observan pocos valores a tipicos pero la matoria tiene que ver con la ditribucion de montos e ingresos por lo que se puede realizar la evaluacion de cada uno para ajustar la variable. Para las variables de saldo y pago se observa que existen muchos valores atipicos que no permiten la visualizacion correcta de los datos. Se recomenda hacer un analisis descriptivo e incluso considerar la agrupacion de esas variables como puede ser el promedio trimestral, entre otros.

# In[249]:


sns.set(rc={'figure.figsize':(15,8)})
sns.boxplot(n_df['X18'])
plt.show()


# 
# ---
# 
# # **Paso 8:Elabora los histogramas de los atributos para visualizar su distribución** 
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[250]:


sns.set(rc={'figure.figsize':(20,40)})
fig, axes = plt.subplots(7,2 )    
  
for k in range(0,14):
  plt.subplot(7,2,k+1) 
  plt.hist(data=n_df, x=columnas_n_df[k])
  plt.xlabel(columnas_n_df[k])  

plt.show()


# 
# ---
# 
# # **Paso 9:** Realiza la visualización de los datos usando por lo menos 3 gráficos que consideres adecuados: plot, scatter, jointplot, boxplot, areaplot, pie chart, pairplot, bar chart, etc.
# 
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# In[276]:


sns.jointplot(data=df1, x='Age', y= 'CreditAmount', hue = 'Gender', palette='dark')


# In[283]:


df1['Age']


# In[287]:


fig, ax1 = plt.subplots() # prepara un gráfico de matplotlib

df1.plot( "Age", "X12", kind="scatter", ax=ax1)

ax1.set_xlabel("X12")
ax1.tick_params(labelsize=20, pad=8)
fig.suptitle('Scatter plot of importe del estado de cuenta de septiembre versus Age', fontsize=15)


# In[ ]:


sns.pairplot(df1, hue="Education")


# In[299]:


sns.pairplot(df1[["X12", 'X13', 'X14']], size=4, aspect=1)


# 
# ---
# 
# # **Paso 10:** Interpreta y explica cada uno de los gráficos indicando cuál es la información más relevante que podría ayudar en el proceso de toma de decisiones.
#  
#   
# 
# 
# ---
# 
# 
# 
# 
# ---
# 
# 

# Se obsreva que los creditos usualmente estan concentrados en montos altos por lo que los montos bajos paren atipicos, tambien que tipo de genero no distingue tal en si otorgan o no el crédito porque se ve que hay proporcion parecida. 
# Sin embargo, se observa que los montos mas bajos se encuentran en hombres y un poco mas altos en mujeres lo que nos dice algo diferente a lo que usualmente sucede en el mercado mexicano, ya que las mujeres del analisis adquieren montos mas altos, es decir, que tienen mas capacidad de pago para adquirir dicha responsabilidad crediticia.
# 
# También se observa que en la mayoria de los saldos del cliente son mas relevantes los saldos mas actuales de las fechas que los saldos de fechas mas lejanas. Donde es importante mencionar que para la dictaminacion de creditos de revisan estatus de pagos y saldos de al menos 6 meses. 
# 
# Ademas se recomienda que se otorguen principalmente creditos a clientes que tengan mas tiempo de buen comportamiento de pago e incluso se les puede ofrecer incrementar el monto del credito.
