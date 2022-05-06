## ======= load modules ======= ##
import glob
import os

import numpy as np
import pandas as pd

from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor


## YOU SHOULD CHECK "data_dir", "subject_dir", "control_csv_name", "case_csv_name"
## A function that changes control/case ratio will be implemented later

##### code example #####
# from DataSetMaker import DataSetMaker
# <- parse some arguments ->
# parser.add_argument("--database",required=True, type=str, choices=['UKB','ABCD'],help='')
# parser.add_argument("--data",default='fmriprep',type=str, choices=['fmriprep','freesurfer'],
#                     help='select ABCD data. fmriprep or freesurfer')
# parser.add_argument("--group_size_adjust", default=False, type=bool, help='if True, make control-case group size 1:1')
# parser.add_argument("--resize", default=80, help='please enter sigle number')
# pasrer.add_argument("--target", required=True, choices=['SuicidalideationPassive','SuicidalideationActive_Ever',
#                    'SuicidalideationActive_PastYear','SelfHarmed_PastYear','SuicideAttempt_Ever','SuicideAttempt_PastYear',
#                    'SuicideTotal'],help='enter target variable')

# <- parse some arguments ->
# args = parser.parse_args()
# datasetMaker = DataSetMaker(args)
# partition = datasetMaker.make_dataset()

seed = 1234
np.random.seed(seed)

class DataSetMaker():
    def __init__(self,args):
        self.args = args
        
        self.ABCD_data = {
                'fmriprep':'/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked',
                'freesurfer':'/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer',
                'FA_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.1.FA_unwarpped_nii',
                'FA_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.2.FA_warpped_nii',
                'MD_unwarpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.3.MD_unwarpped_nii',
                'MD_unwarpped_npy':'/scratch/connectome/3DCNN/data/1.ABCD/3.3.MD_unwarpped_npy',
                'MD_warpped_nii':'/scratch/connectome/3DCNN/data/1.ABCD/3.4.MD_warpped_nii',
                'RD_warpped':'/scratch/connectome/3DCNN/data/1.ABCD/3.5.RD_warpped'
        }
        self.ABCD_subject_dir = '/scratch/connectome/3DCNN/data/1.ABCD/4.demo_qc'
        
        self.UKB_data_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped/'
        self.UKB_subject_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/'
        
        self.target = args.target # 'SuicideAll'
        
    def make_dataset(self):
        self.prepare_data(self.args)
        self.load_images(self.args)
        self.load_control_case_list(self.args)
        self.partition(self.args)
        
        return self.dataset
        
    def prepare_data(self, args):
        global subjectkey
        if args.database =='ABCD':
            self.data_dir = self.ABCD_data_fmriprep_dir if args.data == 'fmriprep'\
                            else self.ABCD_data_freesurfer_dir
            self.subject_dir = self.ABCD_subject_dir
            self.control_csv_name = 'ABCD_suicide_control.csv'
            self.case_csv_name = 'ABCD_suicide_case.csv'
            subjectkey = 'subjectkey'
            
        elif args.database =='UKB':
            self.data_dir = self.UKB_data_dir
            self.subject_dir = self.UKB_subject_dir
            self.csv_name = 'UKB_phenotype.csv'
            subjectkey = 'eid'

        
    ### load image files (subject ID + '.npy') as list
    def load_images(self, args):
        os.chdir(self.data_dir)
        if args.database =='ABCD':
            self.images = pd.Series(sorted(glob.glob('*.npy')))
            self.images_subjectkey = self.images.map(lambda x: x.split('.')[0])    
        elif args.database =='UKB':
            self.images = pd.Series(sorted(glob.glob('*.nii.gz')))
            self.images_subjectkey = self.images.map(lambda x: int(x.split('.')[0]))

        self.image_files = pd.DataFrame({'filename':self.images, subjectkey:self.images_subjectkey})
            
        print("*** Loading image files as list is completed ***")

        
    ### load control & case subjects  & preprocessing
    def load_control_case_list(self, args):
        os.chdir(self.subject_dir)
        if args.database =='ABCD':
            self.control_data = pd.read_csv(self.control_csv_name)
            self.control_data = self.control_data.loc[:,['subjectkey',self.target]]
            self.control_data = self.control_data.sort_values(by='subjectkey')
            self.control_data = self.control_data.dropna(axis = 0) # removing subject have NA values in sex
            self.control_data = self.control_data.reset_index(drop=True) 

            self.case_data = pd.read_csv(self.case_csv_name)
            self.case_data = self.case_data.loc[:,['subjectkey',self.target]]
            self.case_data = self.case_data.sort_values(by='subjectkey')
            self.case_data = self.case_data.dropna(axis = 0) # removing subject have NA values in sex
            self.case_data = self.case_data.reset_index(drop=True) 
            
        elif args.database =='UKB':
            UKB_csv = pd.read_csv(self.csv_name)
            UKB_total = UKB_csv.loc[UKB_csv['eid'].isin(self.images_subjectkey.map(int))]
            UKB_total = UKB_total.sort_values('eid',ignore_index=True)
            
            Suicide_all = UKB_total.loc[(UKB_total.SuicidalideationPassive.isnull() == False) \
                                             | (UKB_total.SuicidalideationActive_Ever.isnull()==False) \
                                             | (UKB_total.SuicidalideationActive_PastYear.isnull()==False) \
                                             | (UKB_total.SuicideAttempt_Ever.isnull()==False) \
                                             | (UKB_total.SuicideAttempt_PastYear.isnull()==False)]
            Suicide_all.reset_index(inplace=True,drop=True)
            
            if self.target == "SuicideTotal":
                self.case_data = Suicide_all[(Suicide_all.SuicidalideationPassive>0) \
                                           | (Suicide_all.SuicidalideationActive_Ever>0) \
                                           | (Suicide_all.SuicidalideationActive_PastYear>0) \
                                           | (Suicide_all.SuicideAttempt_Ever>0) \
                                           | (Suicide_all.SuicideAttempt_PastYear>0)]
                self.control_data = Suicide_all[Suicide_all.eid.isin(self.case_data.eid).map(lambda x: not x)]
            else:
                self.case_data = Suicide_all[Suicide_all[self.target]>0]
                self.control_data = Suicide_all[Suicide_all[self.target]==0]
            
            self.target = self.target+'_bin'
            self.case_data.reset_index(inplace=True, drop=True)
            self.case_data.insert(loc=len(Suicide_all.columns), column=(self.target), value=1)
            
            self.control_data.reset_index(inplace=True, drop=True)
            self.control_data.insert(loc=len(Suicide_all.columns), column=(self.target), value=0)
            
        print("*** Loading control & case subject list is completed ***")
        

    ### make train/test/validation dataset
    def partition(self, args):
        os.chdir(self.data_dir) # if I do not change directory here, image data is not loaded
        # get subject ID and target variables as sorted list
        
        self.case_merged = pd.merge(self.image_files, self.case_data, how='inner',on=subjectkey)
        self.control_merged = pd.merge(self.image_files, self.control_data, how='inner',on=subjectkey)
        
        ## for test -> self.case_merged = self.case_merged[:50]
        
        # make control group size same with case group size    
        if args.group_size_adjust:
            self.control_merged = self.control_merged.loc[:self.case_merged.shape[0]-1]
            # print(self.control_merged.shape[0], self.case_merged.shape[0])
                      
        n_case_total = self.case_merged.shape[0]
        n_case_val = int(n_case_total * args.val_size)
        n_case_test = int(n_case_total * args.test_size)
        n_case_train = n_case_total - (n_case_val+n_case_test)
   
        n_control_total = self.control_merged.shape[0]
        n_control_val = n_case_val
        n_control_test = n_case_test
        n_control_train = n_control_total - (n_control_val+n_control_test)
        
        self.data_info = {'control':[n_control_total, n_control_train, n_control_val, n_control_test],
                          'case':[n_case_total, n_case_train, n_case_val, n_case_test]}
        
        print("\nTotal, train, validation, test dataset size is :")
        print(self.data_info,'\n')

        control_val, control_test, control_train = np.split(self.control_merged, [n_control_val, n_control_val+n_control_test])
        case_val, case_test, case_train = np.split(self.case_merged, [n_case_val, n_case_val+n_case_test])
                          
        self.total_train = control_train.append(case_train,ignore_index=True)
        self.total_val = control_val.append(case_val,ignore_index=True)
        self.total_test = control_test.append(case_test,ignore_index=True)
                                                
        resize = args.resize
        train_transform = Compose([ScaleIntensity(),
                                   AddChannel(),
                                   Resize((int(resize),)*3),
                                  ToTensor()])

        val_transform = Compose([ScaleIntensity(),
                                   AddChannel(),
                                   Resize((int(resize),)*3),
                                  ToTensor()])

        test_transform = Compose([ScaleIntensity(),
                                   AddChannel(),
                                   Resize((int(resize),)*3),
                                  ToTensor()])                                                
        
        train_set = ImageDataset(image_files=self.total_train['filename'],
                                 labels=self.total_train[(self.target)], transform=train_transform)
        val_set = ImageDataset(image_files=self.total_val['filename'],
                               labels=self.total_val[(self.target)], transform=val_transform)
        test_set = ImageDataset(image_files=self.total_test['filename'],
                                labels=self.total_test[(self.target)], transform=test_transform)

        self.dataset = {}
        self.dataset['train'] = train_set
        self.dataset['val'] = val_set
        self.dataset['test'] = test_set
        print("*** Splitting data into train, val, test is completed ***")