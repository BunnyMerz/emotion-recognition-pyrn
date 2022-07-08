import time
class Logger():
    def __init__(self):
        self.train = {}
        self.test = {}

    def add_train(self,epoch,loss,current,total):
        try:
            self.train[epoch]
        except:
            self.train[epoch] = []

        self.train[epoch].append([loss,current,total])

    def add_test(self,epoch,acc,avgloss):
        try:
            self.test[epoch]
        except:
            self.test[epoch] = []

        self.test[epoch].append([acc,avgloss])
        
    def save(self):
        with open(str(int(time.time()))+'.model.txt','w') as log:
            for key in self.train:
                infos = self.train[key]
                log.write("Epoch "+str(key)+'\n')
                for i in infos:
                    train = f"loss:{i[0]},batch:[{i[1]}/{i[2]}]"
                    log.write(train+'\n')
                try:
                    log.write(f"End of epoch. Accuracy:{self.test[key][0]},AvgLoss:{self.test[key][1]}+'\n'")
                except:
                    pass

    def __del__(self):
        del self