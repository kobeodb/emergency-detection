# def crop_sq_bbox(frame, xmin, ymin, xmax, ymax, l_side_length):
#     center_x = (xmin + xmax) / 2
#     center_y = (ymin + ymax) / 2
#
#     width = xmax - xmin
#     height = ymax - ymin
#
#     square_side_length = max(width, height)
#
#     if square_side_length < frame.shape[0] and square_side_length < frame.shape[1]: #The square might fit in the image
#         if (square_side_length / 2) > center_x:
#
