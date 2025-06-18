import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "singapo"))

from generate import generate
from eval_data import EvaluationData
from eval_utils.utils import normalize_mesh,split_by_reference, make_double_sided
from eval_utils.render_utils import project_texture
from eval_utils.render_compare import compare_to_ground_truth
from eval_utils.articulate import articulate

from modules.singapo import SingapoModule
from modules.easitex import EasiTexModule
from modules.projection import ProjectionModule
from modules.module import EvalModule
from modules.texture import TEXTureModule

import argparse
from tqdm import tqdm
import trimesh
import torch
from PIL import Image
import numpy as np


        


def display_results(data, args):
    """
    Display the results of the evaluation.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """


    total_sim = 0
    total_sim_no_tex = 0
    total_sim_naive = 0


    for item in data.get_data_items():
        if item.valid:
            print(f"Item ID: {item.id}")
            print(f"Cosine Similarity: {item.cosine_similarity}")
            print("-" * 20)
            total_sim += item.cosine_similarity
            if args.add_no_texture:
                print(f"Cosine Similarity (No Texture): {item.cosine_similarity_no_easitex}")
                total_sim_no_tex += item.cosine_similarity_no_easitex
                print("-" * 20)
            if args.add_naive_texturing:
                print(f"Cosine Similarity (Naive Projection): {item.naive_cosine_similarity}")
                total_sim_naive += item.naive_cosine_similarity
                print("-" * 20)


    
    print(f"Average Cosine Similarity: {total_sim / len(data.get_data_items())}")
    if args.add_no_texture:
        print(f"Average Cosine Similarity (No Texture): {total_sim_no_tex / len(data.get_data_items())}")
    if args.add_naive_texturing:
        print(f"Average Cosine Similarity (Naive Projection): {total_sim_naive / len(data.get_data_items())}")



def evaluate_meshes(data,args):
    """
    Evaluate the meshes.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Evaluating meshes...")

    for item in tqdm(data.get_data_items()):

        similarity = compare_to_ground_truth(os.path.join(args.from_meshes,item.id,"mesh.obj"),item.obj_path,os.path.join(args.from_meshes,item.id),args)

        item.set_cosine_similarity(similarity)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--singapo_ckpt_path", type=str, default='singapo/exps/singapo/final/ckpts/last.ckpt', help="path to the checkpoint of the model")
    parser.add_argument("--singapo_config_path", type=str, default='singapo/exps/singapo/final/config/parsed.yaml', help="path to the config file")
    parser.add_argument("--singapo_gt_data_root", type=str, default='data', help="the root directory of the original data, used for part mesh retrieval")
    parser.add_argument("--eval_data_path", type=str, default='eval_data', help="path to the data to be evaluated")
    parser.add_argument("--output_path", type=str, default='output', help="path to save the output")
    parser.add_argument("--use_cached", action="store_true", help="whether to use cached objects")
    parser.add_argument("--add_no_texture", action="store_true", help="additionally evaluate the objects without any texturing")
    parser.add_argument("--add_naive_texturing", action="store_true", help="additionally evaluate the objects with naive texturing instead of Easi-Tex")
    parser.add_argument("--from_meshes", type=str, default=None, help="path to the meshes to be evaluated")
    parser.add_argument("--additional_rotations", action="store_true", help="evaluate the objects with additional rotations")
    parser.add_argument("--articulated", action="store_true", help="evaluate objects with articulation")
    parser.add_argument("--add_TEXTure", action="store_true", help="additionally evaluate the objects with TEXTure")

    args = parser.parse_args()

    #All paths used in singapo and easi-tex should be absolute
    args.output_path = os.path.abspath(args.output_path)
    args.singapo_ckpt_path = os.path.abspath(args.singapo_ckpt_path)
    args.singapo_config_path = os.path.abspath(args.singapo_config_path)
    args.singapo_gt_data_root = os.path.abspath(args.singapo_gt_data_root)

    data = EvaluationData(args.eval_data_path,args.output_path,args.use_cached)

    if args.from_meshes is not None:
        print("Evaluating meshes from", args.from_meshes)
        evaluate_meshes(data,args)
        display_results(data,args)
        exit()

    modules = []

    modules.append(SingapoModule(args))
    modules.append(EasiTexModule(args))
    if args.add_naive_texturing:
        modules.append(ProjectionModule(args))
    if args.add_TEXTure:
        modules.append(TEXTureModule(args))
    

    for module in modules:
        module.generate(data)
    
    for module in modules:
        module.evaluate(data)

    
    display_results(data,args)








