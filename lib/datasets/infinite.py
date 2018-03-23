import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
#from voc_eval import voc_eval
from fast_rcnn.config import cfg
import pdb
   
nclasses=13    
class load_data(imdb):    
    
    def __init__(self, path):
        imdb.__init__(self, 'zhwd')
     #   self._devkit_path = self._get_default_path() if devkit_path is None \
    #                        else devkit_path
	print '00000000000000000000000000000000000000000000'
	self._typenum=[]
        self._data_path = path
	#self._type='zhwd'
        #self._classes = ('__background__',self._type)
	#self._nclasses=len(self._classes)
	self._classes=('__background__','zhwd','yhwd',
			'hdfbl','yqdd','zqdd','fdjzzc',
			'yhlt','yqlt','qdfbl','zqlt','zhlt','zqm')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb()
	print '1111111111111111111111111111111111111111111111'
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

     #   assert os.path.exists(self._devkit_path), \
     #           'VOCdevkit path does not exist: {}'.format(self._devkit_path)
       # assert os.path.exists(self._data_path), \
     #           'Path does not exist: {}'.format(self._data_path)    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main','train.txt')
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

       # gt_roidb = [self._load_pascal_annotation(index)
        #            for index in self.image_index]
        gt_roidb=[]
        index=[]
        for i in self._image_index:
            temp=self._load_pascal_annotation(i)
            if temp!=None:
                gt_roidb.append(temp)
                index.append(i)
        self._image_index=index;
	print '))))))))))))))))))))))))))))))))))))))))))))['
	print len(self._image_index)
	#super._image_index=index;       
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        temp=[]
        for i in objs:
            temp.append(i.find('name').text.lower().strip())
        print temp
	#print self._type   #############################################################################################
        #if self._type not in temp:
        #    return  
      #  if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
     #       non_diff_objs = [
   #             obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
         #   objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
	#    if obj.find('name').text.lower().strip()!=self._type:
	#	continue
	    if obj.find('name').text.lower().strip() not in self._typenum:
		self._typenum.append(obj.find('name').text.lower().strip())
		print 'typeoutputlllllllllllllllllllllllllll'
		print self._typenum
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
