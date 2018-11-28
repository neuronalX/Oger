import mdp
import glob
import os
from scipy.io import loadmat

def analog_speech (indir='../datasets/Lyon128'):
    '''
    analog_speech(indir) -> inputs, outputs
    Return data for the isolated digit recognition task (subset of TI46),
    preprocessed using the Lyon Passive ear model. Parameters are:
        - indir: input directory for the data files
    '''
    speech_files = glob.glob(os.path.join(indir, '*.mat'))
    inputs, outputs = [], []
    if len(speech_files) > 0:
        print "Found %d speech_files in directory %s. Loading..." % \
            (len(speech_files), indir)
        #for speech_file in mdp.utils.progressinfo(speech_files):
        for speech_file in speech_files:
            contents = loadmat(speech_file)
            inputs.append(contents['spec'].T)
            outputs.append(-1 * mdp.numx.ones([inputs[-1].shape[0], 10]))
            # Fourth last character in filename indicates the digit class
            outputs[-1][:, speech_file[-5]] = 1
    else:
        print "No speech_files found in %s" % (indir)
        return
    return inputs, outputs

def timit (indir='/afs/elis/group/snn/Oger_datasets/timit4python/53phonemes/train', limit=3696):
    # this code is based on *.mat files for the TIMIT samples
    # - a matlab script to create these files based on TIMIT database is present at ELIS UGent
    # - the phoneme mapping can be defined in the matlab script (dataset train53 uses 53 phonemes)
    files = glob.glob(os.path.join(indir, '*.mat'))
    x, y, samplename = [], [], []
    if len(files) > 0:
        print "Found %d training files in directory %s. Loading %d of them..." % (len(files), indir, limit)
        counter = 0
        for g in mdp.utils.progressinfo(files):
            if counter < limit:
                contents = loadmat(g, struct_as_record=False, mat_dtype=True)
                x.append(contents['data'].T)
                y.append(contents['targets'].T)
                samplename.append(contents['samplename'])
                counter = counter + 1
    else:
        print "No files found in %s" % (indir)
        return
    return x, y, samplename
