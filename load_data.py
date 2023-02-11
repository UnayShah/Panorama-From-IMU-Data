import pickle
import sys
import time

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

def load_data(folder_path, dataset, setname, has_cam, has_vic):

  cfile = folder_path + setname + "/cam/cam" + dataset + ".p"
  ifile = folder_path + setname + "/imu/imuRaw" + dataset + ".p"
  vfile = folder_path + setname + "/vicon/viconRot" + dataset + ".p"

  ts = tic()
  camd = read_data(cfile) if has_cam else None
  imud = read_data(ifile)
  vicd = read_data(vfile) if has_vic else None
  toc(ts,"Data import")
  return camd, imud, vicd