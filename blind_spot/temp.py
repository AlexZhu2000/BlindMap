import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的正弦波图像
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='sin(x)')

# 添加标题和标签
plt.title('Simple Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()

# 添加网格
plt.grid(True)

# 保存图像
plt.savefig('test_plot.png')

# 显示图像
plt.show()

# 清理
plt.close()
import matplotlib
print(matplotlib.get_backend())
import cv2
import numpy as np

# 创建一个空白图像
img = np.zeros((512, 512, 3), np.uint8)
# 显示图像
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)