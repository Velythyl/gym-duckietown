def show_img(np_array):
    from PIL import Image
    img = Image.fromarray(np_array, 'RGB')
    img.show()
