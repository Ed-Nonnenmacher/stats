import numpy as np
import re
import pandas as pd
import time
import math
import nltk,random
from nltk.corpus import names
from nltk.corpus import movie_reviews



e=2.71828182846

##cheating with z-score...
z={0.0: 0.5, 0.01: 0.504, 0.02: 0.508, 0.03: 0.512, 0.04: 0.516, 0.05: 0.5199,
   0.06: 0.5239, 0.07: 0.5279, 0.08: 0.5319, 0.09: 0.5359, 0.1: 0.5398,
   0.11: 0.5438, 0.12: 0.5478, 0.13: 0.5517, 0.14: 0.5557, 0.15: 0.5596,
   0.16: 0.5636, 0.17: 0.5675, 0.18: 0.5714, 0.19: 0.5753, 0.2: 0.5793,
   0.21: 0.5832, 0.22: 0.5871, 0.23: 0.591, 0.24: 0.5948, 0.25: 0.5987,
   0.26: 0.6026, 0.27: 0.6064, 0.28: 0.6103, 0.29: 0.6141, 0.3: 0.6179,
   0.31: 0.6217, 0.32: 0.6255, 0.33: 0.6293, 0.34: 0.6331, 0.35: 0.6368,
   0.36: 0.6406, 0.37: 0.6443, 0.38: 0.648, 0.39: 0.6517, 0.4: 0.6554,
   0.41: 0.6591, 0.42: 0.6628, 0.43: 0.6664, 0.44: 0.67, 0.45: 0.6736,
   0.46: 0.6772, 0.47: 0.6808, 0.48: 0.6844, 0.49: 0.6879, 0.5: 0.6915,
   0.51: 0.695, 0.52: 0.6985, 0.53: 0.7019, 0.54: 0.7054, 0.55: 0.7088,
   0.56: 0.7123, 0.57: 0.7157, 0.58: 0.719, 0.59: 0.7224, 0.6: 0.7257,
   0.61: 0.7291, 0.62: 0.7324, 0.63: 0.7357, 0.64: 0.7389, 0.65: 0.7422,
   0.66: 0.7454, 0.67: 0.7486, 0.68: 0.7517, 0.69: 0.7549, 0.7: 0.758,
   0.71: 0.7611, 0.72: 0.7642, 0.73: 0.7673, 0.74: 0.7704, 0.75: 0.7734,
   0.76: 0.7764, 0.77: 0.7794, 0.78: 0.7823, 0.79: 0.7852, 0.8: 0.7881,
   0.81: 0.791, 0.82: 0.7939, 0.83: 0.7967, 0.84: 0.7995, 0.85: 0.8023,
   0.86: 0.8051, 0.87: 0.8078, 0.88: 0.8106, 0.89: 0.8133, 0.9: 0.8159,
   0.91: 0.8186, 0.92: 0.8212, 0.93: 0.8238, 0.94: 0.8264, 0.95: 0.8289,
   0.96: 0.8315, 0.97: 0.834, 0.98: 0.8365, 0.99: 0.8389, 1.0: 0.8413,
   1.01: 0.8438, 1.02: 0.8461, 1.03: 0.8485, 1.04: 0.8508, 1.05: 0.8531,
   1.06: 0.8554, 1.07: 0.8577, 1.08: 0.8599, 1.09: 0.8621, 1.1: 0.8643,
   1.11: 0.8665, 1.12: 0.8686, 1.13: 0.8708, 1.14: 0.8729, 1.15: 0.8749,
   1.16: 0.877, 1.17: 0.879, 1.18: 0.881, 1.19: 0.883, 1.2: 0.8849,
   1.21: 0.8869, 1.22: 0.8888, 1.23: 0.8907, 1.24: 0.8925, 1.25: 0.8944,
   1.26: 0.8962, 1.27: 0.898, 1.28: 0.8997, 1.29: 0.9015, 1.3: 0.9032,
   1.31: 0.9049, 1.32: 0.9066, 1.33: 0.9082, 1.34: 0.9099, 1.35: 0.9115,
   1.36: 0.9131, 1.37: 0.9147, 1.38: 0.9162, 1.39: 0.9177, 1.4: 0.9192,
   1.41: 0.9207, 1.42: 0.9222, 1.43: 0.9236, 1.44: 0.9251, 1.45: 0.9265,
   1.46: 0.9279, 1.47: 0.9292, 1.48: 0.9306, 1.49: 0.9319, 1.5: 0.9332,
   1.51: 0.9345, 1.52: 0.9357, 1.53: 0.937, 1.54: 0.9382, 1.55: 0.9394,
   1.56: 0.9406, 1.57: 0.9418, 1.58: 0.9429, 1.59: 0.9441, 1.6: 0.9452,
   1.61: 0.9463, 1.62: 0.9474, 1.63: 0.9484, 1.64: 0.9495, 1.65: 0.9505,
   1.66: 0.9515, 1.67: 0.9525, 1.68: 0.9535, 1.69: 0.9545, 1.7: 0.9554,
   1.71: 0.9564, 1.72: 0.9573, 1.73: 0.9582, 1.74: 0.9591, 1.75: 0.9599,
   1.76: 0.9608, 1.77: 0.9616, 1.78: 0.9625, 1.79: 0.9633, 1.8: 0.9641,
   1.81: 0.9649, 1.82: 0.9656, 1.83: 0.9664, 1.84: 0.9671, 1.85: 0.9678,
   1.86: 0.9686, 1.87: 0.9693, 1.88: 0.9699, 1.89: 0.9706, 1.9: 0.9713,
   1.91: 0.9719, 1.92: 0.9726, 1.93: 0.9732, 1.94: 0.9738, 1.95: 0.9744,
   1.96: 0.975, 1.97: 0.9756, 1.98: 0.9761, 1.99: 0.9767, 2.0: 0.9772,
   2.01: 0.9778, 2.02: 0.9783, 2.03: 0.9788, 2.04: 0.9793, 2.05: 0.9798,
   2.06: 0.9803, 2.07: 0.9808, 2.08: 0.9812, 2.09: 0.9817, 2.1: 0.9821,
   2.11: 0.9826, 2.12: 0.983, 2.13: 0.9834, 2.14: 0.9838, 2.15: 0.9842,
   2.16: 0.9846, 2.17: 0.985, 2.18: 0.9854, 2.19: 0.9857, 2.2: 0.9861,
   2.21: 0.9864, 2.22: 0.9868, 2.23: 0.9871, 2.24: 0.9875, 2.25: 0.9878,
   2.26: 0.9881, 2.27: 0.9884, 2.28: 0.9887, 2.29: 0.989, 2.3: 0.9893,
   2.31: 0.9896, 2.32: 0.9898, 2.33: 0.9901, 2.34: 0.9904, 2.35: 0.9906,
   2.36: 0.9909, 2.37: 0.9911, 2.38: 0.9913, 2.39: 0.9916, 2.4: 0.9918,
   2.41: 0.992, 2.42: 0.9922, 2.43: 0.9925, 2.44: 0.9927, 2.45: 0.9929,
   2.46: 0.9931, 2.47: 0.9932, 2.48: 0.9934, 2.49: 0.9936, 2.5: 0.9938,
   2.51: 0.994, 2.52: 0.9941, 2.53: 0.9943, 2.54: 0.9945, 2.55: 0.9946,
   2.56: 0.9948, 2.57: 0.9949, 2.58: 0.9951, 2.59: 0.9952, 2.6: 0.9953,
   2.61: 0.9955, 2.62: 0.9956, 2.63: 0.9957, 2.64: 0.9959, 2.65: 0.996,
   2.66: 0.9961, 2.67: 0.9962, 2.68: 0.9963, 2.69: 0.9964, 2.7: 0.9965,
   2.71: 0.9966, 2.72: 0.9967, 2.73: 0.9968, 2.74: 0.9969, 2.75: 0.997,
   2.76: 0.9971, 2.77: 0.9972, 2.78: 0.9973, 2.79: 0.9974, 2.8: 0.9974,
   2.81: 0.9975, 2.82: 0.9976, 2.83: 0.9977, 2.84: 0.9977, 2.85: 0.9978,
   2.86: 0.9979, 2.87: 0.9979, 2.88: 0.998, 2.89: 0.9981, 2.9: 0.9981,
   2.91: 0.9982, 2.92: 0.9982, 2.93: 0.9983, 2.94: 0.9984, 2.95: 0.9984,
   2.96: 0.9985, 2.97: 0.9985, 2.98: 0.9986, 2.99: 0.9986, 3.0: 0.9987,
   3.01: 0.9987, 3.02: 0.9987, 3.03: 0.9988, 3.04: 0.9988, 3.05: 0.9989,
   3.06: 0.9989, 3.07: 0.9989, 3.08: 0.999, 3.09: 0.999, 3.1: 0.999,
   3.11: 0.9991, 3.12: 0.9991, 3.13: 0.9991, 3.14: 0.9992, 3.15: 0.9992,
   3.16: 0.9992, 3.17: 0.9992, 3.18: 0.9993, 3.19: 0.9993, 3.2: 0.9993,
   3.21: 0.9993, 3.22: 0.9994, 3.23: 0.9994, 3.24: 0.9994, 3.25: 0.9994,
   3.26: 0.9994, 3.27: 0.9995, 3.28: 0.9995, 3.29: 0.9995, 3.3: 0.9995,
   3.31: 0.9995, 3.32: 0.9995, 3.33: 0.9996, 3.34: 0.9996, 3.35: 0.9996,
   3.36: 0.9996, 3.37: 0.9996, 3.38: 0.9996, 3.39: 0.9997, 3.4: 0.9997,
   3.41: 0.9997, 3.42: 0.9997, 3.43: 0.9997, 3.44: 0.9997, 3.45: 0.9997,
   3.46: 0.9997, 3.47: 0.9997, 3.48: 0.9997, 3.49: 0.9998}




min_re=re.compile(r'\d{1,2}\.\d{1,2}')

def freq_dist(stuff):
        '''Frequency Distribution.
        Careful, as the return dict may get huge'''
        items=set(stuff)
        fd=dict()
        for i in items:
                fd[i]=stuff.count(i)
        return fd

def mean(lis):
        return sum(lis)/len(lis)

def variance_s(los):
        '''sample formula (n-1)'''
        var=[(x-mean(los))**2 for x in los]
        return sum(var)/(len(los)-1)
def variance_p(los):
        var=[(x-mean(los))**2 for x in los]
        return sum(var)/(len(los))

def sd_p(lis):
        '''population standard deviation'''
        return variance_p(lis)**(1.0/2)

def sd_s(lis):
        '''sample standard deviation'''
        return variance_s(lis)**(1.0/2)

def sd_stat(stat):
        '''stat is the statistic, can be ppv, npv,spec,sen  | pop=population'''
        return (stat*(1-stat))**(1/2)

def se(stat,sample):
                return ((stat*(1-stat))/sample)**(1/2)

def ppv(a,b):
                '''a=True Positive  b=False Positive'''
                ppv=(a/sum([a,b]))
                ciu=ppv+(1.96*se(ppv,sum([a,b])))
                cil=ppv-(1.96*se(ppv,sum([a,b])))
                return f'{str(round(ppv*100,2))}% ({str(round(cil*100,2))}-{str(round(ciu*100,2))})'


def npv(c,d):
                ''' c=False Negative  d=True Negative'''
                npv=d/sum([c,d])
                ciu=npv+(1.96*se(npv,sum([c,d])))
                cil=npv-(1.96*se(npv,sum([c,d])))
                return f'{str(round(npv*100,2))}% ({str(round(cil*100,2))}-{str(round(ciu*100,2))})'

def sensitivity(a,c):
                sens=a/sum([a,c])
                ciu=sens+(1.96*se(sens,sum([a,c])))
                cil=sens-(1.96*se(sens,sum([a,c])))
                return f'{str(round(sens*100,2))}% ({str(round(cil*100,2))}-{str(round(ciu*100,2))})'

def specificity(b,d):
                '''b=False Positive  D=True Negative'''
                spec=d/sum([b,d])
                ciu=spec+(1.96*se(spec,sum([b,d])))
                cil=spec-(1.96*se(spec,sum([b,d])))
                return f'{str(round(spec*100,2))}% ({str(round(cil*100,2))}-{str(round(ciu*100,2))})'

           
def screen(a,b,c,d):
        return f'{str(ppv(a,b))}\t{str(specificity(b,d))}\t{str(sensitivity(a,c))}'

def factorial(n):
        if n==1:
                return 1
        else:
                return n*factorial(n-1)
                                

        
def n_choose_x(n,x):
        N=factorial(n)
        X=factorial(x)*factorial(n-x)   
        return N/X

def binomial_probability(p,n,x):
        nx=n_choose_x(n,x)
        P=p**x
        p_=(1-p)**(n-x)
        return nx*P*p_



def drt(d=1,r=1,t=1):
        '''Distance = Rate*Time,
                Rate = Distance/Time
                Time = Distance/Rate...
                Function to provide the missing variable in DRT function'''
        if d>1 and r>1:
                m=min_re.search(str(round(float((((d*1000)/r)/60)),2))).group(1)
                s=min_re.search(str(round(float((((d*1000)/r)/60)),2))).group(2)
                print('Time it will take:  '+m+':'+s)
        if d>1 and t>1:
                print('Rate of Speed (D/T):  '+str((d*1000)/t))
        if r>1 and t>1:
                print('Distance (meters):  '+str(r*t))

def median(los):
        if len(los)%2==0:
                los.sort()
                return (los[int((len(los)/2))]+los[int((len(los)/2)-1)])/2,len(los)/2
        else:
                return los[round(len(los)/2)-1],round(len(los)/2)-1
        


def percentile(perc,df,range_=None):
	'''DataFrame must be Series or column you are seeking percentile for
	must be in .iloc[:,0]'''
	if type(df)!=pd.core.frame.DataFrame:
		return 'df must be Pandas DataFrame or Series.'
	else:
		q1=int(round(df.quantile(perc),0))
		if range_==None:
			print(f'Percentile of {perc}, in df is: {q1}')
			print(f'Values less than {q1} that fit within the {perc} percentile range are...')
			return df[df.iloc[:,0]<=q1]
		else:
			q2=int(round(df.quantile(range_),0))
			print(f'Percentile range between {perc} - {range_}, in df is:')
			return df[df.iloc[:,0].between(q1,q2)]
        

def quartile1(los):
        m1=los[:int(median(los)[1])]
        return median(m1)

def quartile3(los):
        m3=los[int(median(los)[1]):]
        return median(m3)

def IQR(los):
        return quartile3(los)[0]-quartile1(los)[0]

def outliers(los):
        iqr=IQR(los)
        return [i for i in los if (i<quartile1(los)[0]-(1.5*iqr) or i>quartile3(los)[0]+(1.5*iqr))],iqr


def OR(a,b,c,d,cross_product=False):
	if cross_product==True:
		or1=a*d
		or2=b*c
		Or=or1/or2
		ciub=e**(math.log(Or)+(1.96*(((1/a)+(1/b)+(1/c)+(1/d))**.5)))
		cilb=e**(math.log(Or)-(1.96*(((1/a)+(1/b)+(1/c)+(1/d))**.5)))
		return Or,cilb,ciub
	p_hat1=a/(a+b)
	p_hat2=c/(c+d)
	odd1=p_hat1/(1-p_hat1)
	odd2=p_hat2/(1-p_hat2)
	Or=odd1/odd2
	ciub=e**(math.log(Or)+(1.96*(((1/a)+(1/b)+(1/c)+(1/d))**.5)))
	cilb=e**(math.log(Or)-(1.96*(((1/a)+(1/b)+(1/c)+(1/d))**.5)))
	return Or,cilb,ciub


def p_hat_CI(response_rate,n,CI):
        p_hat=response_rate/n
        se_p_hat=(p_hat*(1-p_hat)/n)**(1.0/2)
        half_width=CI*se_p_hat
        return p_hat-half_width,p_hat+half_width

def diff_mean(CI,mu_1=None,n1=None,var_1=None,mu_2=None,n2=None,var_2=None,los1=None,los2=None,known_variance=True):
	if known_variance==True:
		mup=mu_1-mu_2
		half_width=CI*((var_1/n1)+(var_2/n2))**(1.0/2)
		return mup-half_width,mup+half_width
	if known_variance==False:
		mu1,n1,var1=mean(los1),len(los1),variance_s(los1)
		mu2,n2,var2=mean(los2),len(los2),variance_s(los2)
		mup=mu1-mu2
		s2p=((n1-1)*var1+(n2-1)*var2)/(n1+n2-2)
		se=(s2p*(1/n1+1/n2))**(1.0/2)
		half_width=CI*se
		print(mup,s2p,se)
		return mup-half_width,mup+half_width

	
def cv_fetch(percent):
	for i in z:
		if z[i]==percent:
			return i




##NOT ACCURATE.  Summation of hypergemoteric function with n choose x is off
'''
class fisher():
	def __init__(self,a,b,c,d):
		self.a=a
		self.b=b
		self.c=c
		self.d=d
	def hyp_geo(self,a,b,c,d):
		base is called on class instantiation
		return (n_choose_x(a+b,c)*n_choose_x(c+d,c)/
			n_choose_x(a+b+c+d,a+c)+(1/n_choose_x(a+b+c+d,a+c)))
'''

##Recursive Markov's chain for probability of tokens
def prob(a):
    if len(a)==1:
        return len([i for i in corpus if a.lower() == i.lower()[0]])/len(corpus)
    else:
        bi2,bi1=a[-1],a[-2]
        rest=a[:-2]
        p=len([i for i in corpus if bi1.lower()+bi2.lower() in i.lower()])/len([i for i in corpus if bi1.lower() in i.lower()])
        return p*prob(rest)




class chi_sq2x2():
	def __init__(self,a,b,c,d):
		self.a=a
		self.b=b
		self.c=c
		self.d=d
		self.n=a+b+c+d
		self.data=[[self.a,self.b],
			   [self.c,self.d]]
		self.expected=[[round((a+b)*(a+c)/self.n,2),
				round((a+b)*(b+d)/self.n,2)],
			       [round((c+d)*(a+c)/self.n,2),
				round((c+d)*(b+d)/self.n,2)]]
		self.test=sum([(self.data[i][x]-self.expected[i][x])**2/
                               self.expected[i][x]
                               for i in range(2)
                               for x in range(2)])
class chi_sq():
	def __init__(self,data):
		self.data=data
		self.n=sum([sum(i) for i in data])
		self.expected=[[(sum(data[row])*sum([z[i] for z in data]))/self.n
                                for i in range(len(data[row]))]
                               for row in range(len(data))]
		self.df=(len(data[0])-1)*(len(data)-1)
		self.test=sum([(self.data[i][x]-self.expected[i][x])**2
			       /self.expected[i][x]
			       for i in range(len(data))
			       for x in range(len(data[i]))])


def sigmoid(z):
        return (e**z)/(e**z+1)


        

class s_lin_reg():
        def __init__(self,x,y):
                self.x=x
                self.y=y
                self.data=[[xx,yy] for xx,yy in zip(x,y)]
                self.xmean=mean(self.x)
                self.ymean=mean(self.y)
                self.sxx=sum([(i[0]-self.xmean)**2 for i in self.data])
                self.syy=sum([(i[1]-self.ymean)**2 for i in self.data])
                self.sxy=sum([(i[0]-self.xmean)*(i[1]-self.ymean) for i in self.data])
                self.slope=self.sxy/self.sxx
                self.y_int=self.ymean-self.slope*self.xmean
                self.ssr=sum([((self.y_int+self.slope*i[0])-self.ymean)**2 for i in self.data])
                self.sse=sum([(i[1]-(self.y_int+self.slope*i[0]))**2 for i in self.data])
                self.sst=self.ssr+self.sse
                
       
                
        
        
##Currently dysfunctional!  Needs some heavy linear algebraic functions
'''
class m_lin_reg():
        def __init__(self,x,y):
                self.x=x
                self.y=y
                self.data=[[yy,xx] for xx,yy in zip(x,y)]
                self.xmeans=[mean(i) for i in self.x]
                self.ymean=mean(self.y)
                self.sxx=[sum((i-xmeans[idx])**2 for i in ii) for idx,ii in enumerate(x)]
                self.syy=sum([(i[1]-self.ymean)**2 for i in self.data])
                self.sxy=sum([(i[0]-self.xmeans)*(i[1]-self.ymean) for i in self.data])
'''


class significance():
        pass
        


def p_value(Z,sides=2):
        '''by default, this is 2 sided'''
        return (1-z[Z])*sides

def t_dist(los,CI):
        mu=mean(los)
        sigma=sd_s(los)
        half_width=CI*(sigma/(len(los)**(1.0/2)))
        return mu-half_width,mu+half_width

def central_limit(x_bar,pop_mean,pop_std,n):
        '''tests whether 2 populations are different
        or not based on hypothesis.  If this value is within
        the null hypothesis (ie. -z 1-a/2< Z < z 1-a/2)'''
        numerator=x_bar-pop_mean
        denom=pop_std/(n**(1.0/2))
        return round(numerator/denom,2)


def record_rng(x,n):
	'''x==number of unique patients
	n==number of total records'''
	days_supply=[30,60,90,120]
	drugs=['A','B','C','D']
	patient=list(range(x))
	return {patient[random.randint(1,x)-1]:
                [datetime.date(random.randint(2000,2019),
                random.randint(1,12),random.randint(1,28)),
                 random.randint(1,30),drugs[random.randint(0,len(drugs)-1)]]
                for i in range(n)}

#--------------------------------
##Algorithms and data structures.
#--------------------------------


##Solid website for optimization and memory management using downscaling: https://www.dataquest.io/blog/pandas-big-data/

df={'WGT': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 21, 7: 21, 8: 18, 9: 18, 10: 1, 11: 1, 12: 3, 13: 3, 14: 2, 15: 2, 16: 1, 17: 1, 18: 1, 19: 1, 20: 6, 21: 6, 22: 4, 23: 4, 24: 1, 25: 1, 26: 2, 27: 2},
    'STRATUM': {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 8, 16: 9, 17: 9, 18: 10, 19: 10, 20: 11, 21: 11, 22: 12, 23: 12, 24: 13, 25: 13, 26: 14, 27: 14},
    'CASE': {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0, 16: 1, 17: 0, 18: 1, 19: 0, 20: 1, 21: 0, 22: 1, 23: 0, 24: 1, 25: 0, 26: 1, 27: 0},
    'EST': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0, 12: 1, 13: 1, 14: 0, 15: 0, 16: 0, 17: 1, 18: 0, 19: 0, 20: 1, 21: 0, 22: 1, 23: 1, 24: 1, 25: 0, 26: 1, 27: 1},
    'GALL': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 0, 13: 1, 14: 1, 15: 0, 16: 1, 17: 0, 18: 1, 19: 1, 20: 1, 21: 0, 22: 1, 23: 0, 24: 1, 25: 1, 26: 1, 27: 1},
    'SURVT': {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2, 8: 1, 9: 2, 10: 1, 11: 2, 12: 1, 13: 2, 14: 1, 15: 2, 16: 1, 17: 2, 18: 1, 19: 2, 20: 1, 21: 2, 22: 1, 23: 2, 24: 1, 25: 2, 26: 1, 27: 2}}


df=pd.DataFrame(df)


def downcast(df):
    df_f=df.select_dtypes([np.float16,np.float64,np.float32])
    df_f=df_f.apply(pd.to_numeric,downcast='float')
    col=list(df_f.columns)
    df_i=df.select_dtypes([np.int32,np.int64])
    df_i=df_i.apply(pd.to_numeric,downcast='integer')
    icol=col+list(df_i.columns)
    rest=[i for i in df.columns if i not in icol]
    rdf=df[rest]
    return df_f.join([df_i,rdf])




def insert(trie,key,value):
    '''generates a lexicon/vocabulary lookup resource.
        Basically dictionaries nested inside more
        dictionaries for every letter of every word'''
    if key:
        first=key[1]
        rest=key[1:]
        if first not in trie:
            trie[first]={}
        insert(trie[first],rest,value)
    else:
        trie['category']=value

def preprocess(tagcorp):
    '''Data structure that stores words as sets, then a list to de-duplicate
    and create a dictionary for fast lookup.... I think'''
    words=set()
    tags=set()
    for sent in tagcorp:
        for word,tag in sent:
            words.add(word)
            tags.add(tag)
    wm=dict((w,i) for (i,w) in enumerate(words))
    tm=dict((t,i) for (i,t) in enumerate(tags))
    return [[(wm[w],tm[t]) for (w,t) in sent] for sent in tagcorp]


def vir(n):
    if n==0:
        return ['']
    elif n==1:
        return ['S']
    else:
        s=['S'+pro for pro in vir(n-1)]
        l=['L'+pro for pro in vir(n-2)]
        return s+l

def binarySearch (arr, l, r, x):
    '''Recursive binary search algorithm'''
    if r >= l:

        mid = int(round(l + (r - l)/2,0))
        
        if arr[mid] == x:
            return mid

        elif arr[mid] > x:
            return binarySearch(arr, l, mid-1, x)


        else:
            return binarySearch(arr, mid+1, r, x)

    else:

        return -1


a=list(range(100))
binarySearch(a,min(a),len(a),30)
##returns index of x


##----------------------DISPROPORTIONALITY SIGNALS--------------------

def RR(a,b,c,d):
    '''Reporting Ratio'''
    try:
        return round((a*(a+b+c+d))/((a+c)*(a+b)),4)
    except (ValueError,ZeroDivisionError):
        return 'N/A'

def PRR(a,b,c,d):
    '''Proportional Reporting Ratio'''
    try:
        return round((a/(a+b))/(c/(c+d)),4)
    except (ValueError,ZeroDivisionError):
        return 'N/A'

def ROR(a,b,c,d):
    '''Reporting Odds Ratio'''
    try:
        return round((a/c)/(b/d),4)
    except (ValueError,ZeroDivisionError):
        return 'N/A'

def IC(a,b,c,d):
    '''Information component.  ln(RR) = logbase2 of reporting ratio'''
    try:
        return round(math.log2(RR(a,b,c,d)),4)
    except:
        return 'N/A'
    
def RR_MGPS(a,b,c,d):
    '''Reporting Ratio for (MGPS) multi-item gamma Poisson shrinker'''
    try:
        return round(a/(((a+b)/(a+b+c+d))*(a+b)),4)
    except (ValueError,ZeroDivisionError):
        return 'N/A'


##Language Processing Feature Extractors.
##Very elementary and serve only as very brief study-guide/reminder =)





def gender_feat1(name):
    return {'last letters':name[-2:],'last':name[-1]}

def gender_feat2(name):
    return {'ct':len(name)}
    
def movie_feat1(doc):
    feat={}
    for i in all_movie_words:
        feat[i]=(i.lower() in all_movie_words)
    return feat


'''
##Name classifier
names=[[i,'male'] for i in names.words('male.txt')]+[[i,'female'] for i in names.words('female.txt')]

name_feats=[[gender_feat1(i[0]),i[1]] for i in names]
random.shuffle(name_feats)
train,test=name_feats[:500],name_feats[500:]

cls=nltk.NaiveBayesClassifier.train(train)
nltk.classify.accuracy(cls,test)



#Movie classifier
all_movie_words=list(nltk.FreqDist(i.lower() for i in movie_reviews.words()))[:2000]##Freq dist of top 2000 most used words in all movie reviews
##there is an additional step that can be done that will help penalize for stop words
movie_docs=[[list(movie_reviews.words(i)),cat]
            for cat in movie_reviews.categories()
            for i in movie_reviews.fileids(cat)]##the corpus


for i in sorted(all_movie_words,key=all_movie_words.__getitem__,reverse=True):
    print(i,all_movie_words[i])
'''
