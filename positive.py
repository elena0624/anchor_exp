"""
Discretizers classes, to be used in lime_tabular
"""
def positive_names(feature_names):
    names={}
    for i in range(len(feature_names)):
        name=feature_names[i]
        names[i]=['%s <0'%name]
        #print(names)
        names[i].append('%s >0'%name)
    return names

def positive_discretizer(data):
    ret = data.copy()
    ret[ret>0]=1
    ret[ret<0]=0
    return ret