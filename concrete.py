import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tkinter as tk

concrete_data=pd.read_csv(r'C:\Users\Tharun\Downloads\concrete_data.csv')
data=pd.DataFrame(concrete_data)
#data.info()
#print(data.describe())
#print(data.isnull().sum())
'''sb.pairplot( data)
plt.figure(figsize=[17,9])
plt.scatter(y='concrete_compressive_strength',x='cement',edgecolors='red',data=data)
plt.ylabel('csMPa')
plt.xlabel('cement')
plt.figure(figsize=[17,8])

sb.heatmap(data.corr(),annot=True)
l=['cement','blast_furnace_slag','fly_ash','water','superplasticizer','coarse_aggregate','age','concrete_compressive_strength']
for i in l:
    sb.boxplot(x=data[i])
    plt.show()'''
x = data.drop(['concrete_compressive_strength'],axis=1)
# dependent variables
y = data['concrete_compressive_strength']

win=tk.Tk()
win.title("window")

lab1=tk.Label(win,text="cement : ").grid(row=0,column=0,sticky=tk.W)
val1=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val1).grid(row=0,column=1)

lab2=tk.Label(win,text="blast_furnace_slag : ").grid(row=1,column=0,sticky=tk.W)
val2=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val2).grid(row=1,column=1)

lab3=tk.Label(win,text="fly_ash : ").grid(row=2,column=0,sticky=tk.W)
val3=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val3).grid(row=2,column=1)

lab4=tk.Label(win,text="water : ").grid(row=3,column=0,sticky=tk.W)
val4=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val4).grid(row=3,column=1)

lab5=tk.Label(win,text="superplasticizer : ").grid(row=4,column=0,sticky=tk.W)
val5=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val5).grid(row=4,column=1)

lab6=tk.Label(win,text="coarse_aggregate : ").grid(row=5,column=0,sticky=tk.W)
val6=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val6).grid(row=5,column=1)

lab7=tk.Label(win,text="fine_aggregate : ").grid(row=6,column=0,sticky=tk.W)
val7=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val7).grid(row=6,column=1)

lab8=tk.Label(win,text="age : ").grid(row=7,column=0,sticky=tk.W)
val8=tk.StringVar()
ent1=tk.Entry(win,width=10,textvariable=val8).grid(row=7,column=1)

def func():
    cement=float(val1.get())
    blast_furnace_slag=float(val2.get())
    fly_ash=float(val3.get())
    water=float(val4.get())
    superplasticizer=float(val5.get())
    coarse_aggregate=float(val6.get())
    fine_aggregate=float(val7.get())
    age=float(val8.get())
    print(cement,blast_furnace_slag,fly_ash,water,superplasticizer,coarse_aggregate,fine_aggregate,age)
    

    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=0)
    print(xtest)
    xtest.loc[len(xtest.index)]=[cement,blast_furnace_slag,fly_ash,water,superplasticizer,coarse_aggregate,fine_aggregate,age]
    print(xtest)
    from sklearn.preprocessing import StandardScaler
    stand= StandardScaler()
    Fit = stand.fit(xtrain)
    xtrain_scl = Fit.transform(xtrain)
    xtest_scl = Fit.transform(xtest)
    last=xtest_scl[-1]
    print(xtest_scl)
    print(last)
    xtest_scl=xtest_scl[:-1]
    print(xtest_scl)
    #xtest.loc[len(xtest.index)]=[1,2,3,4,5,6,7,8]
    #linear regression
    '''from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    lr=LinearRegression()
    fit=lr.fit(xtrain_scl,ytrain)
    score = lr.score(xtest_scl,ytest)
    print(f'predcted score is : {score}')
    print('..................................')
    y_predict = lr.predict(xtest_scl)
    print('mean_sqrd_error is ==',mean_squared_error(ytest,y_predict))
    rms = np.sqrt(mean_squared_error(ytest,y_predict)) 
    print('root mean squared error is == {}'.format(rms))
    plt.figure(figsize=[17,8])
    plt.scatter(y_predict,ytest)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
    plt.xlabel('predicted')
    plt.ylabel('orignal')
    plt.show()'''
    #lasso and rigid
    '''from sklearn.linear_model import Ridge,Lasso
    from sklearn.metrics import mean_squared_error
    rd= Ridge(alpha=0.4)
    ls= Lasso(alpha=0.3)
    fit_rd=rd.fit(xtrain_scl,ytrain)
    fit_ls = ls.fit(xtrain_scl,ytrain)
    print('score od ridge regression is:-',rd.score(xtest_scl,ytest))
    print('.......................................................')
    print('score of lasso is:-',ls.score(xtest_scl,ytest))
    print('mean_sqrd_roor of ridig is==',mean_squared_error(ytest,rd.predict(xtest_scl)))
    print('mean_sqrd_roor of lasso is==',mean_squared_error(ytest,ls.predict(xtest_scl)))
    print('root_mean_squared error of ridge is==',np.sqrt(mean_squared_error(ytest,rd.predict(xtest_scl))))
    plt.figure(figsize=[17,8])
    plt.scatter(rd.predict(xtest_scl),ytest)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
    plt.xlabel('predicted')
    plt.ylabel('orignal')
    plt.show()
    plt.figure(figsize=[17,8])
    plt.scatter(ls.predict(xtest_scl),ytest)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
    plt.xlabel('predicted')
    plt.ylabel('orignal')
    plt.show()'''
    #Random forest regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    rnd= RandomForestRegressor(ccp_alpha=0.0)
    fit_rnd= rnd.fit(xtrain_scl,ytrain)
    print('score is:-',rnd.score(xtest_scl,ytest))
    print('........................................')
    print('mean_sqrd_error is==',mean_squared_error(ytest,rnd.predict(xtest_scl)))
    print('root_mean_squared error of is==',np.sqrt(mean_squared_error(ytest,rnd.predict(xtest_scl))))
    x_predict = list(rnd.predict(xtest_scl))
    predicted_df = {'predicted_values': x_predict, 'original_values': ytest}
    x_predict = list(rnd.predict(xtest_scl))
    predicted_df = {'predicted_values': x_predict, 'original_values': ytest}
    #creating new dataframe
    d=pd.DataFrame(predicted_df)
    print(d.head(20))
    '''plt.figure(figsize=[17,8])
    plt.scatter(rnd.predict(xtest_scl),ytest)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
    plt.xlabel('predicted')
    plt.ylabel('orignal')
    plt.show()'''
    #print(xtest_scl[0])
    #[-0.17003599,0.44523254,-0.81691314,2.18474771,-1.01399523,-0.53277103,-1.26935452,5.17709062]
    print(rnd.predict([last]))
    tk.Label(win,text=f"concrete_compressive_strength is {rnd.predict([last])[0]}").grid(row=9,column=3)
sub=tk.Button(win,text="predict",command=func).grid(row=8,column=0)

win.geometry('500x500')
win.mainloop()