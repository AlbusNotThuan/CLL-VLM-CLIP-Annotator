import sys
sys.path.append('/tmp2/maitanha/vgu/cll_vlm')
from cll_vlm.dataset.caltech101 import Caltech101Dataset
d = Caltech101Dataset(root='/tmp2/maitanha/vgu/cll_vlm/cll_vlm/data/caltech-101')
print('Num samples:', len(d))
print('Num classes:', len(d.classes))
if len(d) > 0:
    print('First item classes:', d.classes[d[0][1]])
