#from . import anchor_base
#import anchor_base
import anchor_code.anchor_base
#from . import anchor_explanation
#import anchor_explanation
import anchor_code.anchor_explanation
#from . import utils
#import utils_1
import anchor_code.utils_1
#import discretize
import anchor_code.discretize
import collections
import sklearn
import numpy as np
import os
import copy
import string
from io import open
import json
import anchor_code.positive#for我的特製函式

def id_generator(size=15):
#    """Helper function to generate random div ids. This is useful for embedding
#    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class AnchorTabularExplainer(object):
    """
    Args:
        class_names: list of strings
        feature_names: list of strings
        data: used to build one hot encoder
        categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal, and thus discretized.
        ordinal_features: list of integers, features that were
    """
    #def __init__(self, class_names, feature_names, data=None,
    def __init__(self, class_names, feature_names, train_data=None, test_data=None,
                 categorical_names=None, ordinal_features=[]):
        """encoder/disc要幹嘛"""
        #init的部分先弄出一個one hot encoder,下面fit的部分再fit出一個min max scaler 但是!!!!!!!!都還沒真的transform
        self.encoder = collections.namedtuple('random_name',
                                              ['transform'])(lambda x: x)
        self.disc = collections.namedtuple('random_name2',
                                              ['discretize'])(lambda x: x)
        self.categorical_features = []
        if categorical_names:
           # TODO: Check if this n_values is correct!!
            cat_names = sorted(categorical_names.keys())
            n_values = [len(categorical_names[i]) for i in cat_names]
            self.encoder = sklearn.preprocessing.OneHotEncoder(
                categorical_features=cat_names,
                n_values=n_values)
            #self.encoder.fit(data)
            self.encoder.fit(train_data)#test###注意 這裡fit的emcoder是會變成one hot "sparse的形式"!!!!!!!
            #self.encoder.fit(test_data)#test只是fit encoder而已 只要train拿來fit test就可以套用了
            self.categorical_features = self.encoder.categorical_features
        if len(ordinal_features) == 0:
            self.ordinal_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names
        self.discretizer ='positive'##testtest

    def fit(self, train_data, train_labels, validation_data,
#            validation_labels, discretizer='quartile'):#正常版
            validation_labels, discretizer='positive'):
        """
        在這裡初始化把很多東西加入explainer裡!! ex. min max std train train_label validation validation_label disc d_train d_validation 離散化後的categorical_names ordinal_features categorical_features
        """ 
        self.min = {}
        self.max = {}
        self.std = {}
        self.train = train_data
        self.train_labels = train_labels
        self.validation = validation_data
        self.validation_labels = validation_labels
        self.scaler = sklearn.preprocessing.StandardScaler()
#        self.scaler.fit(train_data)
        self.scaler.fit(train_data)#test 它只是先做出一個未來要拿來transform的scaler!!!所以現在看不出來
#        self.scaler.fit(test_data)#test test是不是不用做?為什麼??? 不做怎麼放進model李test?
#       把categorical features變成四分衛數分類
        if discretizer == 'quartile':
            #self.disc = discretize.QuartileDiscretizer(train_data, self.categorical_features, self.feature_names)
            self.disc = anchor_code.discretize.QuartileDiscretizer(train_data, self.categorical_features, self.feature_names)
        elif discretizer == 'decile':
            #self.disc = discretize.DecileDiscretizer(train_data, self.categorical_features, self.feature_names)
            self.disc = anchor_code.discretize.DecileDiscretizer(train_data, self.categorical_features, self.feature_names)
        
        #else:#正常版
            #raise ValueError('Discretizer must be quartile or decile')#正常版
        if discretizer =='quartile':#正常的
            self.d_train = self.disc.discretize(self.train)###為什麼這裡又要做一次discretize??不是送進來之前就做過了嗎??? #for正常的discretizer
        elif discretizer=='positive':#特製版
            self.d_train=anchor_code.positive.positive_discretizer(self.train)#test
        #print("train:", self.train)####test
        #print("d_train:", self.d_train)###test 結果它們長的一樣??????為何要再做一次 是不是可以一個是數值化(沒有d)的一個是離散化後(d)的呢 是
        if discretizer =='quartile':#正常的
            self.d_validation = self.disc.discretize(self.validation)####同上疑問 我不是前面做過了嗎 #for正常的discretizer
        elif discretizer=='positive':#特製版
            self.d_validation = anchor_code.positive.positive_discretizer(self.validation)##for我的特製discretizer
        #print("validation:", self.validation)####test
        #print("d_validation:", self.d_validation)###test
        if discretizer =='quartile':#正常的
            val = self.disc.discretize(validation_data)###########################這裡的val是>?????? 跟self.d_validation看起來是一樣的 #for正常的discretizer
        ###########test
        #print("disc.names:",self.disc.names)
        ###########test
        ####以下是我自己加的 因為本來是none type
        if self.categorical_names==None:
            self.categorical_names={}
        ###以上是我自己加的
        if discretizer =='quartile':#正常的
            self.categorical_names.update(self.disc.names)#for正常的discretizer
            self.ordinal_features = [x for x in range(val.shape[1])#for正常的discretizer
                            if x not in self.categorical_features]####這裡應該是 弄清楚這個if~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#for正常的discretizer
        elif discretizer=='positive':#特製版
            self.ordinal_features=[]# for我的特製discretizer
            self.categorical_names=anchor_code.positive.positive_names(self.feature_names)# for我的特製discretizer
        #for我的特製discretizer
        
        for f in range(train_data.shape[1]):
            if f in self.categorical_features and f not in self.ordinal_features:
                continue##所以如果不符合下面三行還會執行嗎?????????????????????????????????????????
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])


    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                          conditions_leq, num_samples, validation=True):
        ####這個函式會從對應的feature rule中sample出符合這些條件的結果(?)
        """
        bla
        """
        train = self.train if not validation else self.validation
        d_train = self.d_train if not validation else self.d_validation
        idx = np.random.choice(range(train.shape[0]), num_samples,
                               replace=True)##########隨機從train裡面挑n個sample replace=true代表可以重複取
        sample = train[idx]###隨機sample出來的那些sample
        d_sample = d_train[idx]####隨機出來並discrete化後的sample
        ###########test以下
        #print("conditions_eq",conditions_eq)
        #print("conditions_geq",conditions_geq)
        #print("conditions_leq",conditions_leq)
        ###########test以上
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)#####sample出的那些sample的f那幾個feature改成現在這個instance的feature(有沒有disctete過?其實這裡不影響!!!因為有discrete化後的conditions_eq都消失了(沒有純categorical類的 都是有ordinal的))
            #不過如果有的話為什麼會這樣寫? 這裡的意思是讓sample的那幾個feature強迫變成那一類
        for f in conditions_geq:#在>的條件feature中
            idx = d_sample[:, f] <= conditions_geq[f]##找出discrete化後的sample中 那幾個feeature<那個instance的 idx=true(1) true的反而好像是不符合條件的
            if f in conditions_leq:##同時該feature<某值且>某值
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)###這行是在??? 如果後半段成立代表True代表這個index也是不符合條件的 +上前面的只要有一個true就會>0 (其實就是=1)
            if idx.sum() == 0: ###有符合條件的東東存在
                continue##所以加這個conitnue幹嘛 如果if不成立也不會怎樣啊
            options = d_train[:, f] > conditions_geq[f]#如果成立 符合那個條件 options有true有fales 如果成立就是true(1)
            if f in conditions_leq:##同上 同時同個f有這個反向條件的
                options = options * (d_train[:, f] <= conditions_leq[f])#如果這個條件也成立的話 true*true就是1的就都符合條件的
            if options.sum() == 0:##沒有符合條件的 
                min_ = conditions_geq.get(f, self.min[f])###get(f,min[f]) 代表他裡面取得conditions_geq[f]或self.min[f] 可是這不合理 因為conditions的rule是discrete的
                max_ = conditions_leq.get(f, self.max[f])###
                to_rep = np.random.uniform(min_, max_, idx.sum())#在min_,max_中間uniform取出idx.sum個
            else:##有符合條件的
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)###從符合條件的train的f中(注意!!是未discrete過的)取出idx.sum()個#replace=true的時候會重複 =false的時候會不重複
            sample[idx, f] = to_rep #sample中不符合規則的f特徵被改為前面sample出來的東西 所以其他feature就維持本來的樣子 
        for f in conditions_leq:#同上
            if f in conditions_geq:#如果裡面有>的 但是因為前面都檢查過了 所以這裡可以略過
                continue
            idx = d_sample[:, f] > conditions_leq[f]#不符合的會=1
            if idx.sum() == 0:#有符合的東西存在
                continue
            options = d_train[:, f] <= conditions_leq[f]#options符合的=1
            if options.sum() == 0:#沒有符合的東西只好隨便sample sample出來的東西是怎麼確保符合條件的?
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)
            sample[idx, f] = to_rep
        return sample


    def transform_to_examples(self, examples, features_in_anchor=[],
                              predicted_label=None):
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        if self.discretizer =='quartile':#正常的
            examples = self.disc.discretize(examples)#for 正常版discretizer
        elif self.discretizer=='positive':#特製版
            examples = anchor_code.positive.positive_discretizer(examples)#for我的特製discretizer
        for ex in examples:
            values = [self.categorical_names[i][int(ex[i])]
                      if i in self.categorical_features
                      else ex[i] for i in range(ex.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        def jsonize(x): return json.dumps(x)
        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {}
            ret['coveredFalse'] = self.transform_to_examples(
                temp['covered_false'], features_in_anchor, predicted_label)
            ret['coveredTrue'] = self.transform_to_examples(
                temp['covered_true'], features_in_anchor, predicted_label)
            ret['uncoveredTrue'] = self.transform_to_examples(
                temp['uncovered_true'], features_in_anchor, predicted_label)
            ret['uncoveredFalse'] = self.transform_to_examples(
                temp['uncovered_false'], features_in_anchor, predicted_label)
            ret['covered'] =self.transform_to_examples(
                temp['covered'], features_in_anchor, predicted_label)
            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}
        weights = [-1 for x in range(instance.shape[0])]
        if self.discretizer =='quartile':#正常的
            instance = self.disc.discretize(exp['instance'].reshape(1, -1))[0]#for正常版discretizer
        elif self.discretizer=='positive':#特製版
            instance = anchor_code.positive.positive_discretizer(exp['instance'].reshape(1, -1))[0]#for我的特製版discretizer
        values = [self.categorical_names[i][int(instance[i])]
                  if i in self.categorical_features
                  else instance[i] for i in range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }
        return ret

    def as_html(self, exp, **kwargs):
        """bla"""
        exp_map = self.to_explanation_map(exp)

        def jsonize(x): return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        return out


    def get_sample_fn(self, data_row, classifier_fn, desired_label=None):#用來獲得sample_fn跟mapping mapping只會做一次 把這個instance的所有rule列出來
        def predict_fn(x):
            #return classifier_fn(self.encoder.transform(x))#在這裡把餵進複雜模型的input正規化(fit的時候算過要如何正規畫)
            return classifier_fn(x)
        print("desired_label:",desired_label)#none
        true_label = desired_label
        if true_label is None:
            true_label = classifier_fn(data_row.reshape(1, -1))[0]
        print("true_label:",true_label)#這裡才會給label 這裡的true label是指跟model predict出來一樣的結果 跟真正的y_train/y_test沒關係(?)
        # must map present here to include categorical features (for conditions_eq), and numerical features for geq and leq
        mapping = {}#這裡完成把現在這個data row的rule列出來的動作
        if self.discretizer =='quartile':#正常的
            data_row = self.disc.discretize(data_row.reshape(1, -1))[0] #如果原本的input不是離散化後的 在這裡離散化 為啥這行註解掉也可以照跑??? 這樣條件不是超嚴苛??? for 正常版discretize
        elif self.discretizer =='positive':#正常的
            data_row = anchor_code.positive.positive_discretizer(data_row.reshape(1, -1))[0] # for我的特製discretizer
        print("data_row:",data_row)###是不是應該要讓他離散化?????上一行註解應拿掉??是的
        print("categorical_feuatres:",self.categorical_features)#test
        ##奇怪我哪裡刪掉了我自己加回去
        if self.discretizer=='quartile':#正常的
            self.categorical_features=list(range(len(self.feature_names)))####for我的特製discretizer
        elif self.discretizer=='positive':#特製版
            self.categorical_features=list(range(len(self.feature_names)))####for我的特製discretizer
        for f in self.categorical_features:##因為離算化後了 所以就是全部的feature(0~38)#奇怪正常版到底哪裡更新categorical_features的 正常的應該也要吧!!!!
            print('f:',f)#test
            if f in self.ordinal_features:##其實也是全部 因為continuos的離散化都變ordinal(0~38)
                for v in range(len(self.categorical_names[f])):##ordinal feature的每個feature被discrete的長度ex.f=0時v就是0,1,2,3,4
                    idx = len(mapping)#現在開始建立mapping 從空空的mapping(0)開始建
                    if data_row[f] <= v and v != len(self.categorical_names[f]) - 1:#如果<=現在的v(不過後面那個條件為什麼要加上去?代表一定都<=最大那類不用說 其實可以前面for多一個-1就好?)
                        mapping[idx] = (f, 'leq', v)
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v)#mapping=> 某個index的資料可以映射成簡單模型的(feature編號,><,值)
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
                    #同一個feature會在mapping裡面對應到多個 例如datarow[0]是1 就會有mapping[0]=(0,geq,0);mapping[1]=(0,leq,1);mapping[2]=(0,leq,2);就這三個而已!!因為最後一個一定成立
            else:
                idx = len(mapping)#其它就是categorical的 所以中間的條件都是=
                mapping[idx] = (f, 'eq', data_row[f])
            # names[idx] = '%s = %s' % (
            #     self.feature_names[f],
            #     self.categorical_names[f][int(data_row[f])])
        print("mapping:",mapping)###test
        ###!!!!因為
        def sample_fn(present, num_samples, compute_labels=True, validation=True):###present是指現在的sample的feature rule(對應到mapping會有100多個的) 但他到底哪裡知道present是什麼的>??????離散後的編號 把他對應成conditions eq leq geq
            #這個函式是用來 讓取出的feature rule index對應成他的條件
            conditions_eq = {}#for categorical features
            conditions_leq = {}#<
            conditions_geq = {}#>
            print("present:",present)#################present代表的是現在有哪些feautre rule被加進anchor
            for x in present:#在這些feature rule裡面
                f, op, v = mapping[x]
                if op == 'eq':#當它是=的時候
                    conditions_eq[f] = v#它的值是v
                if op == 'leq':#當它是<的時候
                    if f not in conditions_leq:#如果該feature還沒被加到conditions裡 (why要加這句?可能同時有多個feature rule用一樣的f?如果前面有就不更新 如果前面沒有就直接加新的)
                        conditions_leq[f] = v#它的值是v
                    conditions_leq[f] = min(conditions_leq[f], v)#有了這條前面是不是就不用if了(?)=> 前面的if保證在執行這個比較的時候conditions_leq[f]不會是空值 這個代表 有心的f跟v 跟既有的f跟v比 如果新的比較小就變新的 如果就得比較小就維持舊的 因為<是要找比較小的 不然可能會變成>
                if op == 'geq':#當它是>的時候
                    if f not in conditions_geq:#那個feature
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            #test
            #print("conditions_eq",conditions_eq)
            #print("conditions_leq",conditions_leq)
            #print("conditions_geq",conditions_geq)
            #test
            # conditions_eq = dict([(x, data_row[x]) for x in present])
            ##做完以上把rule對應成條件後 就可以去sample了!!詳見sample_from_train
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples,
                validation=validation)
            if self.discretizer =='quartile':#正常的
                d_raw_data = self.disc.discretize(raw_data)######把sample出來的raw data分成discrete化後的結果# for正常版discretizer
            elif self.discretizer=='positive':#特製版
                d_raw_data = anchor_code.positive.positive_discretizer(raw_data)######for我的特製版discretizer
            data = np.zeros((num_samples, len(mapping)), int)#data=num_samplesx全部規則的個數 代表確定每個sample符合條件的情形(1or0)
            for i in mapping:#為什麼這裡要再做一次 前面符合條見的sample from traiN不就確保sample出來的東西都會符合條件了嗎 nono 這裡的len(mapping是有全部的條件 包括不在present裡的) 
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)#data第i個rule
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            ####test
            #print("predict_fn(raw_data):",predict_fn(raw_data))
            #print("labels:",labels)
            #####test
            return raw_data, data, labels#這裡的labels指的是跟原預測一不一樣 是fidelity
        return sample_fn, mapping

    def explain_instance(self, data_row, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                          max_anchor_size=None,
                          desired_label=None,
                          beam_size=4, **kwargs):
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, desired_label=desired_label)#送進去的有原始的instance model_predict
        # return sample_fn, mapping
#        exp = anchor_base.AnchorBaseBeam.anchor_beam(
        exp = anchor_code.anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, max_anchor_size=max_anchor_size,
            **kwargs)
        self.add_names_to_exp(data_row, exp, mapping)
        exp['instance'] = data_row
        exp['prediction'] = classifier_fn(data_row.reshape(1,-1))[0].astype(int)
#       exp['prediction'] = classifier_fn(self.encoder.transform(data_row.reshape(1, -1)))[0]
        #explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        explanation = anchor_code.anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        return explanation

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        # TODO: precision recall is all wrong, coverage functions wont work
        # anymore due to ranges
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            # v = data_row[f]
            if op == 'eq':#沒有這個選項應該不會進來
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                    #fname = '%s' % (fname, self.categorical_names[f][v]) #test 
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        #geq_val = name.split()[0]
                        geq_val = "".join(name.split()[0:2])
                    elif '>' in name:
                        #geq_val = name.split()[-1]
                        geq_val = "".join(name.split()[-2:])
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        #leq_val = name.split()[-1]
                        leq_val = "".join(name.split()[-2:])#test
                    elif '<' in name:
                        #leq_val = name.split()[-1]
                        leq_val = "".join(name.split()[-2:])
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
#                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
#                    fname = '%s %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(fname)
