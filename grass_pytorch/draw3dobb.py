import matplotlib.pyplot as plt
import matplotlib.patches as patches

mat_data = loadmat('out.mat')

init_obb = mat_data['init_obb']
gt_reg = mat_data['gt_reg']
predict_reg = mat_data['predict_reg']


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.add_patch(
    patches.Rectangle(
        (0.1, 0.1),   # (x,y)
        0.5,          # width
        0.5,          # height
    )
)
fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')