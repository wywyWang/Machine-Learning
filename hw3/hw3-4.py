import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math
# Import data
filename='Concrete_Data.csv'
my_data = pd.read_csv(filename,encoding="utf-8")

from sklearn.model_selection import train_test_split
print(my_data.columns[0:])

#my_data

#my_data
from sklearn import preprocessing 
my_data = preprocessing.scale(my_data)
train, test = train_test_split(my_data, test_size=0.2)

print(len(train))
print(len(test))


train_feature = train[:,0:8]
# print(train_feature)


train_feature_x = train[:,0:1]
train_feature_y = train[:,1:2]
train_feature_z = train[:,2:3]
train_feature_a = train[:,3:4]
train_feature_b = train[:,4:5]
train_feature_c = train[:,5:6]
train_feature_d = train[:,6:7]
train_feature_e = train[:,7:8]


train_feature_xx = train[:,2:3]
train_feature_yy = train[:,3:4]
train_feature_zz = train[:,2:3]
train_feature_aa = train[:,4:5]
train_feature_bb = train[:,2:3]
train_feature_cc = train[:,3:4]
train_feature_dd = train[:,4:5]
train_feature_ee = train[:,2:3]

train_feature_xy = train[:,2:3]
train_feature_xz = train[:,3:4]
train_feature_xa = train[:,3:4]
train_feature_xb = train[:,4:5]
train_feature_xc = train[:,2:3]
train_feature_xd = train[:,3:4]
train_feature_xe = train[:,4:5]
train_feature_yz = train[:,2:3]
train_feature_ya = train[:,2:3]
train_feature_yb = train[:,3:4]
train_feature_yc = train[:,3:4]
train_feature_yd = train[:,4:5]
train_feature_ye = train[:,2:3]
train_feature_za = train[:,3:4]
train_feature_zb = train[:,4:5]
train_feature_zc = train[:,2:3]
train_feature_zd = train[:,2:3]
train_feature_ze = train[:,3:4]
train_feature_ab = train[:,3:4]
train_feature_ac = train[:,4:5]
train_feature_ad = train[:,2:3]
train_feature_ae = train[:,3:4]
train_feature_bc = train[:,4:5]
train_feature_bd = train[:,2:3]
train_feature_be = train[:,4:5]
train_feature_cd = train[:,2:3]
train_feature_ce = train[:,3:4]
train_feature_de = train[:,4:5]

print("train_feature_x")
# print(train_feature_x[0])


for i in range(len(train_feature_xx)):
    train_feature_xx[i] = train_feature_x[i]*train_feature_x[i]
    train_feature_yy[i] = train_feature_y[i]*train_feature_y[i]
    train_feature_zz[i] = train_feature_z[i]*train_feature_z[i]
    train_feature_aa[i] = train_feature_a[i]*train_feature_a[i]
    train_feature_bb[i] = train_feature_b[i]*train_feature_b[i]
    train_feature_cc[i] = train_feature_c[i]*train_feature_c[i]
    train_feature_dd[i] = train_feature_d[i]*train_feature_d[i]
    train_feature_ee[i] = train_feature_e[i]*train_feature_e[i]
    train_feature_xy[i] = train_feature_x[i]*train_feature_y[i]
    train_feature_xz[i] = train_feature_x[i]*train_feature_z[i]
    train_feature_xa[i] = train_feature_x[i]*train_feature_a[i]
    train_feature_xb[i] = train_feature_x[i]*train_feature_b[i]
    train_feature_xc[i] = train_feature_x[i]*train_feature_c[i]
    train_feature_xd[i] = train_feature_x[i]*train_feature_d[i]
    train_feature_xe[i] = train_feature_x[i]*train_feature_e[i]
    train_feature_yz[i] = train_feature_y[i]*train_feature_z[i]
    train_feature_ya[i] = train_feature_y[i]*train_feature_a[i]
    train_feature_yb[i] = train_feature_y[i]*train_feature_b[i]
    train_feature_yc[i] = train_feature_y[i]*train_feature_c[i]
    train_feature_yd[i] = train_feature_y[i]*train_feature_d[i]
    train_feature_ye[i] = train_feature_y[i]*train_feature_e[i]
    train_feature_za[i] = train_feature_z[i]*train_feature_a[i]
    train_feature_zb[i] = train_feature_z[i]*train_feature_b[i]
    train_feature_zc[i] = train_feature_z[i]*train_feature_c[i]
    train_feature_zd[i] = train_feature_z[i]*train_feature_d[i]
    train_feature_ze[i] = train_feature_z[i]*train_feature_e[i]
    train_feature_ab[i] = train_feature_a[i]*train_feature_b[i]
    train_feature_ac[i] = train_feature_a[i]*train_feature_c[i]
    train_feature_ad[i] = train_feature_a[i]*train_feature_d[i]
    train_feature_ae[i] = train_feature_a[i]*train_feature_e[i]
    train_feature_bc[i] = train_feature_b[i]*train_feature_c[i]
    train_feature_bd[i] = train_feature_b[i]*train_feature_d[i]
    train_feature_be[i] = train_feature_b[i]*train_feature_e[i]
    train_feature_cd[i] = train_feature_c[i]*train_feature_d[i]
    train_feature_ce[i] = train_feature_c[i]*train_feature_e[i]
    train_feature_de[i] = train_feature_d[i]*train_feature_e[i]


ones = np.ones([train_feature.shape[0], 1])
#print(ones)

train_feature = np.concatenate((ones, train_feature), axis=1)

train_feature = np.concatenate((train_feature,train_feature_xx), axis=1)
train_feature = np.concatenate((train_feature,train_feature_yy), axis=1)
train_feature = np.concatenate((train_feature,train_feature_zz), axis=1)
train_feature = np.concatenate((train_feature,train_feature_aa), axis=1)
train_feature = np.concatenate((train_feature,train_feature_bb), axis=1)
train_feature = np.concatenate((train_feature,train_feature_cc), axis=1)
train_feature = np.concatenate((train_feature,train_feature_dd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ee), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xy), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xz), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xa), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xb), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xc), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_xe), axis=1)
train_feature = np.concatenate((train_feature,train_feature_yz), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ya), axis=1)
train_feature = np.concatenate((train_feature,train_feature_yb), axis=1)
train_feature = np.concatenate((train_feature,train_feature_yc), axis=1)
train_feature = np.concatenate((train_feature,train_feature_yd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ye), axis=1)
train_feature = np.concatenate((train_feature,train_feature_za), axis=1)
train_feature = np.concatenate((train_feature,train_feature_zb), axis=1)
train_feature = np.concatenate((train_feature,train_feature_zc), axis=1)
train_feature = np.concatenate((train_feature,train_feature_zd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ze), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ab), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ac), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ad), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ae), axis=1)
train_feature = np.concatenate((train_feature,train_feature_bc), axis=1)
train_feature = np.concatenate((train_feature,train_feature_bd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_be), axis=1)
train_feature = np.concatenate((train_feature,train_feature_cd), axis=1)
train_feature = np.concatenate((train_feature,train_feature_ce), axis=1)
train_feature = np.concatenate((train_feature,train_feature_de), axis=1)


train_label = train[:,8:9]
train_label_bar = train_label.mean()
test_label = test[:,8:9]

theta = np.array([0.002 * np.random.random_sample(train_feature.shape[1]) - 0.001])
# theta=np.full((1, 45), 0.4)
print(theta)
alpha = 100
iters = 100000

m_t = 0
v_t = 0
iters_count = 0

def gradientDescent(train_feature, train_label, theta, iters_count, alpha, m_t, v_t):
    while(iters_count < 50000):
        iters_count += 1
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8
        print(train_feature.shape)
        #print(np.sum(train_feature@theta.T, axis = 1))
        #r2_now = r2_score(train_label, train_feature@theta.T);
        y = np.dot(train_feature, theta.T).reshape(train_feature.shape[0], 1)
        gt = (-1 * (train_label - y).T.dot(train_feature).reshape(train_feature.shape[1]) / train_feature.shape[0])
        m_t = beta_1*m_t + (1-beta_1)*gt
        v_t = beta_2*v_t + (1-beta_2)*gt*gt
        m_cap = m_t/(1-(beta_1))
        
        v_cap = v_t/(1-(beta_2))
        theta_0_prev = theta

        a = (alpha*m_cap)/(np.sqrt(v_cap)+eps)
        theta = theta - a
        if(np.equal(theta,theta_0_prev).all()):
            break
        #theta = theta - (alpha/len(train_feature)) * np.sum(train_feature*(train_feature@theta.T - train_label), axis=0)
        print("iteration = ",iters_count)
        #print("R2: ",r2_now)
        print("value is : ",theta)
    return theta

final_theta = gradientDescent(train_feature,train_label,theta[0],iters_count,alpha, m_t, v_t)
print(final_theta)


test_feature = test[:,0:8]
# print(test_feature)


test_feature_x = test[:,0:1]
test_feature_y = test[:,1:2]
test_feature_z = test[:,2:3]
test_feature_a = test[:,3:4]
test_feature_b = test[:,4:5]
test_feature_c = test[:,5:6]
test_feature_d = test[:,6:7]
test_feature_e = test[:,7:8]


test_feature_xx = test[:,2:3]
test_feature_yy = test[:,3:4]
test_feature_zz = test[:,2:3]
test_feature_aa = test[:,4:5]
test_feature_bb = test[:,2:3]
test_feature_cc = test[:,3:4]
test_feature_dd = test[:,4:5]
test_feature_ee = test[:,2:3]
                  
test_feature_xy = test[:,2:3]
test_feature_xz = test[:,3:4]
test_feature_xa = test[:,3:4]
test_feature_xb = test[:,4:5]
test_feature_xc = test[:,2:3]
test_feature_xd = test[:,3:4]
test_feature_xe = test[:,4:5]
test_feature_yz = test[:,2:3]
test_feature_ya = test[:,2:3]
test_feature_yb = test[:,3:4]
test_feature_yc = test[:,3:4]
test_feature_yd = test[:,4:5]
test_feature_ye = test[:,2:3]
test_feature_za = test[:,3:4]
test_feature_zb = test[:,4:5]
test_feature_zc = test[:,2:3]
test_feature_zd = test[:,2:3]
test_feature_ze = test[:,3:4]
test_feature_ab = test[:,3:4]
test_feature_ac = test[:,4:5]
test_feature_ad = test[:,2:3]
test_feature_ae = test[:,3:4]
test_feature_bc = test[:,4:5]
test_feature_bd = test[:,2:3]
test_feature_be = test[:,4:5]
test_feature_cd = test[:,2:3]
test_feature_ce = test[:,3:4]
test_feature_de = test[:,4:5]

for i in range(len(test_feature_x)):
    test_feature_xx[i] = test_feature_x[i]*test_feature_x[i]
    test_feature_yy[i] = test_feature_y[i]*test_feature_y[i]
    test_feature_zz[i] = test_feature_z[i]*test_feature_z[i]
    test_feature_aa[i] = test_feature_a[i]*test_feature_a[i]
    test_feature_bb[i] = test_feature_b[i]*test_feature_b[i]
    test_feature_cc[i] = test_feature_c[i]*test_feature_c[i]
    test_feature_dd[i] = test_feature_d[i]*test_feature_d[i]
    test_feature_ee[i] = test_feature_e[i]*test_feature_e[i]
    test_feature_xy[i] = test_feature_x[i]*test_feature_y[i]
    test_feature_xz[i] = test_feature_x[i]*test_feature_z[i]
    test_feature_xa[i] = test_feature_x[i]*test_feature_a[i]
    test_feature_xb[i] = test_feature_x[i]*test_feature_b[i]
    test_feature_xc[i] = test_feature_x[i]*test_feature_c[i]
    test_feature_xd[i] = test_feature_x[i]*test_feature_d[i]
    test_feature_xe[i] = test_feature_x[i]*test_feature_e[i]
    test_feature_yz[i] = test_feature_y[i]*test_feature_z[i]
    test_feature_ya[i] = test_feature_y[i]*test_feature_a[i]
    test_feature_yb[i] = test_feature_y[i]*test_feature_b[i]
    test_feature_yc[i] = test_feature_y[i]*test_feature_c[i]
    test_feature_yd[i] = test_feature_y[i]*test_feature_d[i]
    test_feature_ye[i] = test_feature_y[i]*test_feature_e[i]
    test_feature_za[i] = test_feature_z[i]*test_feature_a[i]
    test_feature_zb[i] = test_feature_z[i]*test_feature_b[i]
    test_feature_zc[i] = test_feature_z[i]*test_feature_c[i]
    test_feature_zd[i] = test_feature_z[i]*test_feature_d[i]
    test_feature_ze[i] = test_feature_z[i]*test_feature_e[i]
    test_feature_ab[i] = test_feature_a[i]*test_feature_b[i]
    test_feature_ac[i] = test_feature_a[i]*test_feature_c[i]
    test_feature_ad[i] = test_feature_a[i]*test_feature_d[i]
    test_feature_ae[i] = test_feature_a[i]*test_feature_e[i]
    test_feature_bc[i] = test_feature_b[i]*test_feature_c[i]
    test_feature_bd[i] = test_feature_b[i]*test_feature_d[i]
    test_feature_be[i] = test_feature_b[i]*test_feature_e[i]
    test_feature_cd[i] = test_feature_c[i]*test_feature_d[i]
    test_feature_ce[i] = test_feature_c[i]*test_feature_e[i]
    test_feature_de[i] = test_feature_d[i]*test_feature_e[i]


ones = np.ones([test_feature.shape[0], 1])
#print(ones)

test_feature = np.concatenate((ones, test_feature), axis=1)

test_feature = np.concatenate((test_feature,test_feature_xx), axis=1)
test_feature = np.concatenate((test_feature,test_feature_yy), axis=1)
test_feature = np.concatenate((test_feature,test_feature_zz), axis=1)
test_feature = np.concatenate((test_feature,test_feature_aa), axis=1)
test_feature = np.concatenate((test_feature,test_feature_bb), axis=1)
test_feature = np.concatenate((test_feature,test_feature_cc), axis=1)
test_feature = np.concatenate((test_feature,test_feature_dd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ee), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xy), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xz), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xa), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xb), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xc), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_xe), axis=1)
test_feature = np.concatenate((test_feature,test_feature_yz), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ya), axis=1)
test_feature = np.concatenate((test_feature,test_feature_yb), axis=1)
test_feature = np.concatenate((test_feature,test_feature_yc), axis=1)
test_feature = np.concatenate((test_feature,test_feature_yd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ye), axis=1)
test_feature = np.concatenate((test_feature,test_feature_za), axis=1)
test_feature = np.concatenate((test_feature,test_feature_zb), axis=1)
test_feature = np.concatenate((test_feature,test_feature_zc), axis=1)
test_feature = np.concatenate((test_feature,test_feature_zd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ze), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ab), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ac), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ad), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ae), axis=1)
test_feature = np.concatenate((test_feature,test_feature_bc), axis=1)
test_feature = np.concatenate((test_feature,test_feature_bd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_be), axis=1)
test_feature = np.concatenate((test_feature,test_feature_cd), axis=1)
test_feature = np.concatenate((test_feature,test_feature_ce), axis=1)
test_feature = np.concatenate((test_feature,test_feature_de), axis=1)




print(test_label[0:10])
print((test_feature @ final_theta.T)[0:10])


def computeCost_res(feature,label,theta):
    tobesummed = np.power((predict_test_label-label), 2)
    return np.sum(tobesummed)/len(feature)

def computeCost_total(label,test_label_bar,theta):
    tobesummed2 = np.power((label - test_label_bar),2)
    return np.sum(tobesummed2)/len(label)

test_label_bar = test_label.mean()



predict_test_label = final_theta[0] + final_theta[1] * test_feature_x + final_theta[2] * test_feature_y  + \
                     final_theta[3] * test_feature_z + final_theta[4] * test_feature_a + \
                     final_theta[5] * test_feature_b + final_theta[6] * test_feature_c + \
                     final_theta[7] * test_feature_d + final_theta[8] * test_feature_e + \
                     final_theta[9] * test_feature_xx + final_theta[10] * test_feature_yy + \
                     final_theta[11] * test_feature_zz + final_theta[12] * test_feature_aa + \
                     final_theta[13] * test_feature_bb + final_theta[14] * test_feature_cc + \
                     final_theta[15] * test_feature_dd + final_theta[16] * test_feature_ee + \
                     final_theta[17] * test_feature_xy + final_theta[18] * test_feature_xz + \
                     final_theta[19] * test_feature_xa + final_theta[20] * test_feature_xb + \
                     final_theta[21] * test_feature_xc + final_theta[22] * test_feature_xd + \
                     final_theta[23] * test_feature_xe + final_theta[24] * test_feature_yz + \
                     final_theta[25] * test_feature_ya + final_theta[26] * test_feature_yb + \
                     final_theta[27] * test_feature_yc + final_theta[28] * test_feature_yd + \
                     final_theta[29] * test_feature_ye + final_theta[30] * test_feature_za + \
                     final_theta[31] * test_feature_zb + final_theta[32] * test_feature_zc + \
                     final_theta[33] * test_feature_zd + final_theta[34] * test_feature_ze + \
                     final_theta[35] * test_feature_ab + final_theta[36] * test_feature_ac + \
                     final_theta[37] * test_feature_ad + final_theta[38] * test_feature_ae + \
                     final_theta[39] * test_feature_bc + final_theta[40] * test_feature_bd + \
                     final_theta[41] * test_feature_be + final_theta[42] * test_feature_cd + \
                     final_theta[43] * test_feature_ce + final_theta[44] * test_feature_de

    
test_label = test[:,8:9]
R2_from_sklearn = r2_score(test_label,predict_test_label)
print(R2_from_sklearn)
print(final_theta)
print(test_label_bar)
print(computeCost_res(test_feature, test_label, final_theta))

print(computeCost_total(test_label, test_label_bar, final_theta))

R2 = 1- computeCost_res(predict_test_label, test_label, final_theta) / computeCost_total(test_label, test_label_bar, final_theta)
print("R2 = ",R2)

def CountMSE(test_feature, test_label, bias, final_theta):
    se = 0.0
    for i in range(len(test_feature)):
        se += (test_label[i]-np.sum(test_feature[i]*final_theta[0].T+bias)) ** 2
    return se/float(len(test_feature))

mse = CountMSE(test_feature, test_label,computeCost_res(test_feature,test_label,final_theta),final_theta)
print(mse[0])

