import matplotlib.pyplot as plt

workingMonths=[6,17,18,24,30,36,40,48,56,60,62,64,69,70,76,80,82,86,90,96]
salary=[40,52,53,55,55,60,63,66,66,67,68,68,68,72,73,79,81,81,81,90]


plt.scatter(workingMonths, salary,c="blue")
plt.title("Working month- Salary")
plt.xlabel("Working month")
plt.ylabel("Salary")

m=0.1
b=2

predicted_salary=[ x*m+b for x in  workingMonths ]
lossValue=sum([(y-y_pred)**2 for y,y_pred in zip(salary,predicted_salary) ])

def getGradients(x,y,m,b):
    n=len(x)
    mDiff=sum([_x*(_y-((m*_x)+b)) for _x,_y in zip(x,y)])
    bDiff=sum([_y-((m*_x)+b) for _x,_y  in zip(x,y)])
    mGradient=-(2/n)*mDiff
    bGradient=-(2/n)*bDiff
    return (mGradient,bGradient)

predicted_salary2=[]
learningRate=0.0001

for i in range(50000):
    _m,_b=getGradients(workingMonths,salary,m,b)
    m-=learningRate*_m
    b-=learningRate*_b 
    predicted_salary2=[ x*m+b for x in  workingMonths ]
    lossValue2=sum([(y-y_pred)**2 for y,y_pred in zip(salary,predicted_salary2) ])


plt.plot(workingMonths, predicted_salary2,c="red")

plt.show()