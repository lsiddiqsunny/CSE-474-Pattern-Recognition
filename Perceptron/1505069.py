import numpy as np

features=0
classes=0
samples=0

def data_loader(filename,isTrainData):
    # open data file
    file = open("Data_files/Lab#2/"+filename,"r")


    # initialize
    i=0
    global features
    global classes
    global samples


    listx = []
    listy = []

    for line in file:
        # for the first line
        if(i==0 and isTrainData==1):
            fields = line.split()

            features = int(fields[0])
            classes = int(fields[1])
            samples = int(fields[2])
        # for the rest of the line
        else:
            fields = line.split()
            templist = []

            for j in range(features):
                #print(fields[j])
                templist.append(float(fields[j]))

            listx.append(templist)
            listy.append(int(fields[features]))

        i = i+1

    #print(str(features)+" "+str(classes)+" "+str(samples))

    # convert into numpy array
    x = np.array(listx)
    y = np.array(listy)

    #print(x.shape, y.shape)
    
    return x,y

def basic_parceptron(train_x,train_y):
    print('\n\n------------Basic parceptron------------\n\n')
    np.random.seed(1)
    w = np.zeros(features)
    print('Initial weight: ',w)
    bias = np.zeros(1)
    print('Initial bias: ',bias)
    iteration = 0
    max_iteration = 500
    learning_rate = 0.1
    #loss_rate = []

    while True :
        if iteration>=max_iteration :
            break
        iteration = iteration + 1
        current_pred  = train_y.copy()
        for i in range(samples):

            value = np.dot(w,train_x[i])+bias[0]
            #print(value)
            if value>=0:
                current_pred[i] = 1
            else :
                current_pred[i] = 2
        false = 0
        #print(current_pred)
        for i in range(samples):
            if current_pred[i] == train_y[i]:
                continue
            else:
                #print(current_pred[i])
                false = false+1
                for j in range(features):
                    if current_pred[i] == 1:
                        w[j] = w[j]+learning_rate*-1*train_x[i][j]
                    else:
                        w[j] = w[j]+learning_rate*1*train_x[i][j]
                if current_pred[i] == 1:
                    bias[0] = bias[0]+learning_rate*-1*1
                else:
                    bias[0] = bias[0]+learning_rate*1*1
        #loss_rate.append((false/samples)*100)
        #print(false)
        if false == 0:
            break

    print('Iteration:'+str(iteration))
    print('Calculated weight: ',w)
    print('Calculated bias: ',bias)
    return w,bias

def reward_panishment(train_x,train_y):
    print('\n\n------------Reward and Panishment------------\n\n')
    np.random.seed(1)
    w = np.zeros(features)
    print('Initial weight: ',w)
    bias = np.zeros(1)
    print('Initial bias: ',bias)

    iteration = 0
    max_iteration = 5000
    learning_rate = 0.1
    #loss_rate = []
    unchanged = 0
    while True :
        if iteration>=max_iteration :
            break
        iteration = iteration + 1
        for i in range(samples):
            value = 0.0
            for j in range(features):
                value = np.dot(w,train_x[i])+bias[0]
                
            choosedclass = -1
            if value>=0:
                choosedclass = 1
            else :
                choosedclass = 2
                
            if choosedclass == train_y[i]:
                unchanged = unchanged +1
            else:
                unchanged = 0
                for j in range(features):
                    if choosedclass == 1:
                        w[j] = w[j]-learning_rate*1*train_x[i][j]
                    else:
                        w[j] = w[j]-learning_rate*-1*train_x[i][j]
                if choosedclass == 1:
                    bias[0] = bias[0]-learning_rate*1*1
                else:
                    bias[0] = bias[0]-learning_rate*-1*1
            if unchanged == samples:
                break
        if unchanged == samples:
            break
    print('Iteration:'+str(iteration))
    print('Calculated weight: ',w)
    print('Calculated bias: ',bias)
    return w,bias
def pocket(train_x,train_y):
    print('\n\n------------Pocket Algorithm------------\n\n')

    np.random.seed(100)
    w = np.zeros(features)
    print('Initial weight: ',w)
    ws = w.copy()
    bias = np.zeros(1)
    print('Initial bias: ',bias)
    bs = bias.copy()
    hs=0

    iteration = 0
    max_iteration = 500
    learning_rate = 1
    #loss_rate = []

    while True :
        if iteration>=max_iteration :
            break
            
        iteration = iteration + 1
        current_pred  = train_y.copy()
        
        for i in range(samples):
            value = bias[0]
            for j in range(features):
                value = value + w[j]*train_x[i][j]
            #print(value)
            if value>=0:
                current_pred[i] = 1
            else :
                current_pred[i] = 2
                
        correct = 0
        for i in range(samples):
            if current_pred[i] == train_y[i]:
                correct = correct + 1
    
        if correct > hs :
            ws = w.copy()
            bs = bias.copy()
            hs = correct
        if correct == samples:
            break    
        
            
        for i in range(samples):
            if current_pred[i] == train_y[i]:
                continue
            else:
                #print(current_pred[i])
                for j in range(features):
                    if current_pred[i] == 1:
                        w[j] = w[j]-learning_rate*1*train_x[i][j]
                    else:
                        w[j] = w[j]-learning_rate*-1*train_x[i][j]
                if current_pred[i] == 1:
                    bias[0] = bias[0]-learning_rate*1*1
                else:
                    bias[0] = bias[0]-learning_rate*-1*1


        

    w = ws.copy()
    bias = bs.copy()
    print('Iteration:'+str(iteration))
    print('Calculated weight: ',w)
    print('Calculated bias: ',bias)
    return w,bias

def test_result(w,bias,test_x,test_y):
    correct = 0
    current_pred  = test_y.copy()
    for i in range(samples):
        for j in range(features):
            value = np.dot(w,test_x[i])+bias[0]
        if value>=0:
            current_pred[i] = 1
        else :
            current_pred[i] = 2
    for i in range(samples):
        if current_pred[i] == test_y[i]:
            correct = correct + 1
        else:
            print("Test sample no #"+str(i+1)+" "+str(test_x[i])+" "+str(current_pred[i])+" "+str(test_y[i]))

    print("Accuracy : "+str((correct/samples)*100))

def main():
    train_x,train_y =  data_loader("trainLinearlySeparable.txt",1)
    w,bias = basic_parceptron(train_x,train_y)
    test_x,test_y =  data_loader("testLinearlySeparable.txt",0)
    test_result(w,bias,test_x,test_y)
    w,bias = reward_panishment(train_x,train_y)
    test_result(w,bias,test_x,test_y)

    train_x,train_y =  data_loader("trainLinearlyNonSeparable.txt",1)
    w,bias = pocket(train_x,train_y)
    test_x,test_y =  data_loader("testLinearlyNonSeparable.txt",0)
    test_result(w,bias,test_x,test_y)


if __name__ == "__main__":
    main()