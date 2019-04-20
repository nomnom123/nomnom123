#PCA unsupervised learning

Projection of M1 on z1 is X1 cos theta
Projection of M2 on z1 is X2 sin theta
Projection of M1 on z2 is -X1 sin theta
Projection of M2 on z2 is X2 cos theta

z1 =  x1 cos theta + x2 sin theta.........(i)
z2 = -x1 sin theta + x2 cos theta.........(ii)

|z1|=|cos theta   sin theta|.|x1|
|z2|=|-sin theta  cos theta| |x2|
  z=A^t.x.................................(iii)

So, 

  |   a11         a12   |
a=|cos theta   sin theta|
  |  a21         a22    |
  |-sin theta  cos theta|
  
For P variable case we will write eq (iii) as,

|z1| = |a11,a12,...,a1P|   .  |x1|
|z2|   |a21,a22,...,a2P|      |x2|
|. |   |.              |      |. |
|. |   |.              |      |. |
|zP|   |aP1,aP2,...,aPP|PxP   |xP|Px1

Let,
         a1          a2
A = |cos theta  -sin theta|
    |sin theta   cos theta|

a1 = |cos theta|
     |sin theta| 

a2 = |-sin theta|
     | cos theta|
  
a1.a1^T = |cos theta|  |cos theta . sin theta|
          |sin theta| 
        = cos^2 theta +  sin^2 theta = 1

Similarly,
a2.a2^T = 1

A^T.A = | cos theta  sin theta| . |cos theta  -sin theta|
        |-sin theta  cos theta|   |sin theta   cos theta|
   
      = |1  0| = 1.......................(iv)
        |0  1|
  
z = A^T.x

Orthogonal transformation

z1 = a1^Tx = a11x1 + a12x2 + ... + a1PxP
z2 = a2^Tx = a21x1 + a22x2 + ... + a2PxP
.
.
.
zj = aj^Tx = aj1x1 + aj2x2 + ... + ajPxP

zP = aP^Tx = aP1x1 + aP2x2 + ... + aPPxP

Subject to a^Ta = 1 and var(z2) to be minimum

var(z1) >= var(z2) >= ... >= var(zP)

zj = aj^Tx
var(zj) = var(aj^Tx)
        = aj^T var(x)aj..........[vr(ax) = a^2var(x)].............(v)

var(x) is co-variance of all x

let, cov(x) = sigma symbol
cov(x) = sigma symbol =  |sigma11  sigma12 ... sigma 1P| 
                         |sigma21  sigma22 ... sigma 2P| 
                         |sigmaP1  sigmaP2 ... sigma PP| PxP..............(vi)

Mean of ith component,
E(zj) =  E(aj^T.x)
      =  aj^T E(x)
      =  aj^T .mu.........................................(vii)

Therefore,
zj ~ N(aj^TN, aj^T var(x)aj)..............................(viii)

For sample S,
zj ~ N(aj^TN, aj^T Saj)

Conditions:
  
1. each Pi is a linear combination of x.
2. The first PC will capture the maximum variability var (a^Tx)
     Max theta = var(z1)
     Subject to a^T.a = 1
   And,
     Max of remaining var(z2),
     Subject to a2^Ta2 = 1 and cov (a1^Tx, a2^Tx) = 0
   Therefore,
     Max var(zi) = aj^T S.aj
     Subject to aj^T aj = 1.................................(ix)
    Or,
     aj^T.aj-1 = 0
    L=aj^T Saj - lambda(aj^T aj - 1)
    
dL/daj = 0
z aj S - z lambda aj = 0..... {cut z on both the sides}
aj S = lambda aj..............(Eigen values and Eigen vectors)
aj(S - lambda) = 0
Therefore,
  |S - lambda I| = 0......................................(x)
where, 
  lambda = Eigen value
  aj     = Eigen matrix
  
 
#LDA Z-score
  
vector = V^T.x

Calculate mean for x1 and x2
In matrix form (muhat1-muhat2) is represented as
  (muhat1-muhat2).(muhat1-muhat2)^T
Therefore,
  (muhat1-muhat2)^2 = (muhat1-muhat2).(muhat1-muhat2)^T
Now,
  muhat1 = mu1 V^T
Therefore,
  (mu1 V^T - mu2 V^T)^2 = (V^T mu1 - V^T mu2).(V^T mu1 - V^T mu2)^T
                        = V^T(mu1 -  mu2).(mu1 - mu2)^T.V
                        = V^T.SB.V
where,
  SB = (mu1 -  mu2).(mu1 - mu2)^T
Now,
S1hat^2 = sigma(xi - xdash)^2
S1hat^2 = sigma;n2;i=1 (V^T.xi - V^Tmu1)^2
Similarly,
S2hat^2 = sigma;n2;i=1 (V^T.xi - V^Tmu2)^2

Now,
S1hat^2 = V^T(xi-mu1)(xi-mu1)^T.V
        = V^T.SW.V
where
SW = (xi-mu1)(xi-mu1)^T

Now, cost of 
V,
J(V) = V^T.SB.V/V^T.S1.V+V^T.S2.V
J(V) = V^T.SB.V/V^T.SW.V

Differentiating J(V)wrt V,

d(J(V))/d(V) = d/dV. (V^T.SB.V)(V^T.SW.V)-(V^T.SB.V) d/dV (V^T.SW.V)
              ------------------------------------------------------
                                  (V^T.SW.V)^2

             = (2 SB V)V^T.SW.V - V^T.SB.V(2 V SW)
                ----------------------------------- = 0
                          (V^T.SW.V)^2 

             = (V^T.SW.V). SB V  -  V^T.SB.V.SW.V
               ----------------    --------------  = 0
                   V^T.SW.V            V^T.SW.V
                   
SB.V - V^T.SB.V  . SW.V
       --------         =  0
       V^T.SW.V
       
Now,

V^T.SB.V  
--------  =  lambda symbol
V^T.SW.V

Therefore,
  SB.V - lambda SW.V = 0
  SB.V = lambda SW.V
  SB SW^-1 V = lambda V
  SB SW^-1 = A
  A.V = lambda V
  |A - lambda I| = 0 ..................(eigen vector)
  
#LDA GAusian normal distribution
  
According to Bayes Theoram,

P(A/B) = P(A).P(B/A)
        ------------
            P(B)


P(AI/B) = P(AI).P(B/AI)
          ------------
              P(B)

Now, consider two classes i and j and let X represent various parameters to find conclusions or probability of i and j
Consider,
  if, P(i/x) > P(j/x)
  then x<-i
  else x<-j
  
Therefore,
  P(i/x) > P(j/x)

  P(i).P(i/x)   P(j).P(j/x)
  ----------  > -----------
      P(x)          P(x)
  
  P(i).P(i/x) >= P(j).P(j/x)......................(i)
  
Now, P(i) and P(j) both are prior probabilities
Consider P(i) and P(j) both as 0.5,
By Gaussian Distribution,
  P(x/i) =        1
          -----------------* e^-(x-mu)^2/2 sigma^2.............(ii)
           sqrt(2 pi) sigma
           
For multivariate Gaussian,
  P(xi/i) =        1
            -----------------* e^-(x-mu i)^T.ci^-1(x-mu i)/2.............(iii)
            sqrt(2 pi) |ci|

  P(x/j) = {replace i with j in(iii)}..........................(iv)
            
Substituting values of (iii) and (iv) in (i),
P(i).P(i/x) >= P(i).P(j/x)

P(i).   1    
     -------------- * e^-(x-mu i)^T.(i^-1(x- mu i))/2   >= similarly write for p(j)
    sqrt(2 pi) |ci|
      
    {cancel 1/sqrt 2 pi and again write the entire eq}
    
Now, taking log on both sides,

-1/2 log ci + log Pi - 1/2 (x-mu i)^T. ci^-1(x-mu i) >= similarly write for j

Multiply with -2,
 
log ci -2 log Pi + (x-mu i)^T ci^-1(x-mu i) >= similarly write for j...............(i)

Consider ci=cj=ci i.e., Gaussian matrix,
For LDA, eq (i) will be,

log c -2 log Pi + (x-mu i)^T.c^-1(x-mu j)

Assume that it is a symmetric matrix,
(x - mu i)^T c^-1 (x-mu i)
  = x c^-1 x^T - mu i^T c^-1 x - 2mu i c^-1 x^T <= similarly write for j

log Pi + mu i c^-1 x^T - 1/2 mu i c^-1 x^T = f1
similarly write for j                      = f2

f1>f2

##Gradient descent (Logistic Regression)

h(xi) = y = theta 0 + theta 1 xi
J(theta 0, theta 1) = sigmoid symbol (theta 0 + theta 1 xi)............(cost function)
we need to minimize the cost function

y hat = 1/1-e^-(theta 0 + theta 1 xi).............(sigmoid function)

for logistic regression, cost function is defined as,
  
  J(theta 0, theta 1) = -[y log y hat + (1-y) log (1-y hat)]
                        -1/m summation;m;i=1 [y log y hat + (1-y) log (1-y hat)]

theta 0|
       |--y = theta 0 + theta 1 xi --> y hat = 1/1+e^-y --> J(theta 0, theta 1)
theta 1|
  
Now,
 dy/d theta 0 = 1...................(i)
 dy/d theta 1 = xi..................(ii)
 dy hat/dy = (1/1+e^-y)(1-1/1+e^-y) = y hat(1-y hat)...............(iii)
 
dJ(theta 0, theta 1)/dy hat = -[y/y hat + (1-y/1-y hat)(-1)]
dJ/dy hat = -y/y hat + (1-y/1-y hat)..........................(iv)
 
By using eq (i), (iii), (iv),
 dy/d theta 0 * dy hat/dy * dJ/dy hat = 1 * y hat(1-y hat)[-y/y hat + (1-y/1-y hat)]

dJ/theta 0 = (1-y hat)(-y) + y hat(1-y)
           = -y + y.y hat + y hat - y.y.hat.........{cut y hat}
          = y hat - y............(v)

Now, using eq (ii), (iii) and (iv),
 dy/d theta 1 * dy hat/dy * dJ/dy hat = xi.y hat(1-y hat).[-y/y hat + (1-y/1-y hat)]
 dJ/d theta 0 = (y hat - y)xi................(vi)
 
Now, theta 0, theta 1,
theta 0 = theta 0 - alpha.dJ/d theta 0
theta 1 = theta 1 - alpha.dJ/d theta 1.................(alpha = learning parameter)

##Gradient descent - linear regression

h(x) = theta 0 + theta 1.x...............(i)
As per linear regression,
y = alpha + beya1.x
let alpha = theta 0, beta = theta 1
y=h(x)
J(theta) is the cost function
To minimize J(theta),
J(theta) = 1/m summation;m;i=1 (h(xi)-yi)^2
m=no of obs
Min J(theta 0, theta 1)= 1/2m summation;m;i=1 (h(xi)-yi)^2
                       = 1/2m summation;m;i=1 (theta 0 + theta 1*1 - yi)^2
J(theta) = 1/2m summation;m;i=1 (theta 1.xi- yi)^2
         = 1/2m summation;m;i=1 (h(xi) - yi)^2.................(ii)

Differentiating J(theta 0, theta 1) wrt theta 1,
dJ(theta 0, theta 1)/dJ(theta 0) = 1/m summation;m;i=1 (h(xi) - yi)..........(iii)
dJ(theta 0, theta 1)/dJ(theta 1) = 1/m summation;m;i=1 (h(xi) - yi)xi ..........(iv)

Gradient descent:-
  now, theta 0= theta 0 - alpha.dJ(theta 0, theta 1)/dJ(theta 0)
  now, theta 1= theta 1 - alpha.dJ(theta 0, theta 1)/dJ(theta 1)
  
##svm
  
Vector x = (x1,x2,x3,...,xn)
MAgnitude of vector x, ||x||
By euclidean distance norm formula,
||x|| = sqrt(x1^2, x2^2,...,xn^2).............(i)
Direction of the vector mu = (mu1, mu2) is the vector,

       mu1        mu2
W = (------- ,  -------).............(ii)
      ||W||      ||W||
  
{graph}: cos theta= mu1/||mu||
         cos alpha = mu2/||mu||..............(iii)

The line equation,
y = b + ax  or y = mx + c
Now, take the two dimenasional vectors
x = (x1, x2)  and w=(a,-1)
  wx + b = 0......................(iv)
  
Now, w^T.x = [a,-1]|x1|
                   |x2|
           = ax1-x2
wax + w1y + b = 0
w1y = -wax - b
y = -w0/w1.x-b/w1
      or
x2= -w0/w1.x1 + b/w1..............(v)

For +ve class, wx + b >= 0;+1
For -ve class, wx + b <= 0;-1...........(vi)

Multiply with yi,
yi(wx + b).............(vii)

Now, v = |v1|
         |v2|
         |. |
         |vn|
  
||v|| = sqrt(v1^2+v2^2+...+vn^2)
      = sqrt(v.v)................(dot product)
{graph}
Now, wx + b = 0 at x
So, w = x + b = 0
Now, k = d.W/||W||
  x dash = x-k
         = x - d.W/||W||.............(viii)
wx dash + b = 0
W[x - dW/||W||]+b = 0
Wx - dW^2/||W|| +b = 0
Wx - dW ||W|| +b = 0
d =  wx + b/||W||...................(ix)
d = (W/||W||).x + b/||W||
  
Objective to find minimum d
yi((W/||W||).x + b/||W||) = d1..............(x)
d1 = min yi (wx + b/||W||)
for both sides,
2d1 = min yi(wx + b/||W||)
Now minimize 2d1 subject to,
yi[(wx + b/||W||)]>=d1..................(from vii)
||W|| = 1/d1
So, max 2.1/||W||
subject to,
yi[(wx + b/||W||)]>= 1/||W||
  
Min 1/2||W||
subject to,
yi[Wxi + b]>= 1....................(xi)






    
            

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

          
  
  


  





