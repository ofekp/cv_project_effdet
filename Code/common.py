def get_image_path(main_folder_path, image_id, is_colab):
    if is_colab:
        return main_folder_path + f'/../Data/busesTrain/DSCF{image_id}.JPG'
    else:
        return main_folder_path + f'/../Data/busesTrain/DSCF{image_id}.JPG'
