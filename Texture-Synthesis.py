# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:10:00 2021

@author: hatam
"""
import numpy as np
import argparse
import cv2
import os
import ffmpeg
inf = np.float('inf')
output_size=2500
def find_patch_first_row(template, source,blocksize):
    template=template.astype('uint8')
    
    res = cv2.matchTemplate(source[:,0: len(source[0])-blocksize],template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.90
    loc = np.where(res >= threshold )
    #print(len(loc[1]))
    while len(loc[1])<100:
        threshold=threshold-0.03
        loc = np.where(res >= threshold )
        
    h_rand = np.random.randint(len(loc[1]))
    return source[loc[0][h_rand]:loc[0][h_rand]+blocksize,loc[1][h_rand]:loc[1][h_rand]+blocksize,:] 

def find_patch_first_col(template, source,blocksize):
    template=template.astype('uint8')
    
    res = cv2.matchTemplate(source[0: len(source)-blocksize,:],template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.90
    loc = np.where(res >= threshold )
    #print(len(loc[1]))
    while len(loc[1])<100:
        threshold=threshold-0.03
        loc = np.where(res >= threshold )
        
    h_rand = np.random.randint(len(loc[1]))
    res=source[loc[0][h_rand]:loc[0][h_rand]+blocksize,loc[1][h_rand]:loc[1][h_rand]+blocksize,:]
    #print(res.shape)
    return source[loc[0][h_rand]:loc[0][h_rand]+blocksize,loc[1][h_rand]:loc[1][h_rand]+blocksize,:] 

def find_patch_others(blktop,blkleft , source,blocksize, overlap):
    
    template1=np.zeros((overlap,blocksize,3))
    template2=np.zeros((blocksize,overlap,3))
    template1=blktop
    template2=blkleft
    template1=template1.astype('uint8')
    template2=template2.astype('uint8')
    #print(template1.shape)
    res1 = cv2.matchTemplate(source[:len(source)-blocksize,: len(source[0])-blocksize,:],template1,cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(source[:len(source)-blocksize,: len(source[0])-blocksize,:],template2,cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc1 = np.where(res1 >= threshold )
    #print(len(loc1[1]))
    while len(loc1[1])<300:
        threshold=threshold-0.01
        loc1 = np.where(res1 >= threshold )
        #print(len(loc1[1]))
    threshold = 0.80
    loc2 = np.where(res2 >= threshold )
   #print(len(loc2[1]))
    while len(loc2[1])<300:
        threshold=threshold-0.01
        loc2 = np.where(res2 >= threshold )
        #print(len(loc2[1]))
    x=np.where((loc1[0]==loc2[0]))
    h_rand = np.random.randint(len(x))
    res=source[loc1[0][h_rand]:loc1[0][h_rand]+blocksize,loc1[1][h_rand]:loc1[1][h_rand]+blocksize,:]
    #print(res.shape)
    return res
def mincut_first_row(blk1,blk2,blocksize,overlap):
    error=((blk1[:,-overlap:  ]-blk2[:, :overlap])**2).mean(2)
   
    min_index=[]
    energy_final=[list(error[0])]


    for i in range (0,blocksize-1):
        #assing a big number to first and last 
        energy= [inf] + energy_final[-1] + [inf]
        e=np.array([energy[:-2],energy[1:-1],energy[2:]])
        min_energy=e.min(0)
        energy_final_i = error[i+1] + min_energy
        energy_final.append(list(energy_final_i))
        
        min_arrg=e.argmin(0)-1
        min_index.append(min_arrg)
        
        # minimum energy of last element
        min_cut = []
        min_arrg = np.argmin(energy_final[-1])
        
        min_cut.append(min_arrg)
  
    for i in min_index[::-1]:
        min_arrg = min_arrg + i[min_arrg]
        min_cut.append(min_arrg)
        

	# Reverse to find full path
    min_cut = min_cut[::-1]
    mask = np.zeros((blocksize, blocksize, 3))
    for i in range(len(min_cut)):
            mask[i, :min_cut[i]+1] = 1


    return mask
            
def  mincut_first_col(blk1,blk2,blocksize,overlap):
	mask = mincut_first_row(np.rot90(blk1,1), np.rot90(blk2,1), blocksize, overlap)
	return np.rot90(mask,k=3)

def mincut_others(ref_block_top,ref_block_left,added_patch,blocksize,overlap):
    mask1=mincut_first_row(ref_block_left,added_patch,blocksize,overlap)
    mask2=mincut_first_col(ref_block_top,added_patch,blocksize,overlap)
    #print("a",mask1.shape)
    #print(mask2.shape)
    mask=np.logical_or(mask1,mask2)
    
    
    return mask
def assign_pixel(patch, blocksize,overlap):
    
    
    final_image=np.zeros((output_size,output_size,3)) 
    #first block
    h,w= patch.shape[:2]
    #print(H,W)
    
    h_rand = np.random.randint(h - blocksize)
    w_rand = np.random.randint(w - blocksize)
    
    startBlock = patch[h_rand:h_rand+blocksize, w_rand:w_rand+blocksize]
    final_image[0:blocksize,0 :blocksize, :] = startBlock
    
    
    counter=0
    #first row in fimal image
    num_row=0
    num_of_block=(output_size-blocksize)/(blocksize-overlap)
    for i in range(0,int(num_of_block)):
        endpoint=blocksize+num_row*(blocksize-overlap)
        refrance_block=final_image[0:blocksize,endpoint-overlap :endpoint, :]
        ref_block=final_image[0:blocksize,endpoint-blocksize :endpoint, :]
        #print(ref_block.shape)
        added_patch=find_patch_first_row(refrance_block,patch,blocksize)
        #print(added_patch.shape)
        
        #know we should find mincut
        
        mask=mincut_first_row(ref_block,added_patch,blocksize,overlap)
        resblock = np.zeros((blocksize,blocksize,3))
        resblock[:, :overlap] = ref_block[:, -overlap:]
        resblock = resblock*mask + added_patch*(1-mask)
        final_image[:blocksize,((num_row+1)*(blocksize-overlap)):((num_row+1)*(blocksize-overlap)+blocksize),:] = resblock
        num_row=num_row+1
        '''
        counter+=1
        filename = "gif3/file-%d.png"%(counter)
        cv2.imwrite(filename, final_image)'''
        
    print("fist row ...")   
        
     #first column in fimal image
    num_col=0
    num_of_block=(output_size-blocksize)/(blocksize-overlap)
    for i in range(0,int(num_of_block)):
        endpoint=blocksize+num_col*(blocksize-overlap)
        refrance_block=final_image[endpoint-overlap :endpoint,0:blocksize, :]
        ref_block=final_image[endpoint-blocksize :endpoint,0:blocksize, :]
        #print(ref_block.shape)
        added_patch=find_patch_first_col(refrance_block,patch,blocksize)
        #print(added_patch.shape)
        #know we should find mincut
        
        mask=mincut_first_col(ref_block,added_patch,blocksize,overlap)
        resblock = np.zeros((blocksize,blocksize,3))
        resblock[:overlap, :] = ref_block[-overlap:,:]
        #cv2.imwrite("texture3.jpg", final_image)
        resblock = resblock*mask + added_patch*(1-mask)
        final_image[((num_col+1)*(blocksize-overlap)):((num_col+1)*(blocksize-overlap)+blocksize),:blocksize,:] = resblock
        num_col=num_col+1 
        #cv2.imwrite("texture3.jpg", final_image)
        '''
        counter+=1
        filename = "gif3/file-%d.png"%(counter)
        cv2.imwrite(filename, final_image)'''
    
    print("fist col ...") 
    ####others rows and column
    
    print("others rows and column....") 
    num_of_block=(output_size-blocksize)/(blocksize-overlap)
    for num_row in range(1,int(num_of_block)+1):
        #num_col=1
        for num_col in range(1,int(num_of_block)+1):
            endpoint_h=blocksize+num_row*(blocksize-overlap)
            endpoint_w=blocksize+num_col*(blocksize-overlap)
            ref_block_top=final_image[endpoint_h-blocksize :endpoint_h-blocksize+overlap,endpoint_w-blocksize:endpoint_w, :]
            ref_block_left=final_image[endpoint_h-blocksize:endpoint_h,endpoint_w-blocksize :endpoint_w-blocksize+overlap, :]
            #print("  a",ref_block_top.shape,ref_block_left.shape)
            cv2.imwrite("left.jpg",ref_block_left)
            cv2.imwrite("top.jpg",ref_block_top)
            added_patch=find_patch_others(ref_block_top,ref_block_left , patch,blocksize, overlap)
            #final_image[endpoint_w-blocksize:endpoint_w,endpoint_h-blocksize:endpoint_h,:] = added_patch
            
            mask=mincut_others(ref_block_top,ref_block_left,added_patch,blocksize,overlap)
            resblock = np.zeros((blocksize,blocksize,3))
            resblock[:overlap, :] = ref_block_top
            resblock[:, :overlap] = ref_block_left
            
            #cv2.imwrite("texture3.jpg", final_image)
            resblock = resblock*mask + added_patch*(1-mask)
            final_image[endpoint_h-blocksize:endpoint_h,endpoint_w-blocksize :endpoint_w,:] = resblock
            '''
            counter+=1
            filename = "gif3/file-%d.png"%(counter)
            cv2.imwrite(filename, final_image)'''
                
            
   
        cv2.imwrite("res1.jpg", final_image)
  
    



#read image
patch=cv2.imread("texture1.jpg")

blocksize=100
overlap=40
assign_pixel(patch, blocksize,overlap)


#os.system('ffmpeg -i gif3/file-%d.png -r 10 -vcodec mpeg4 contourgif30.MP4')

