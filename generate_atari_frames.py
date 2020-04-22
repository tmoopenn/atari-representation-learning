"""
Script for generating frames from atari environment using either a random or pretrained_agent 
and organizing them into a dataset 
"""
import argparse
import numpy as np
import os
import re
from atariari.benchmark.episodes import get_random_agent_rollouts, get_ppo_rollouts 
from PIL import Image

'''
Checkpoint Indexes for pretrained ppo agent 
checkpointed_steps_full_sorted = [1536, 1076736, 2151936, 3227136, 4302336, 5377536, 6452736, 7527936, 8603136, 9678336,
                                  10753536, 11828736, 12903936, 13979136, 15054336, 16129536, 17204736, 18279936,
                                  19355136, 20430336, 21505536, 22580736, 23655936, 24731136, 25806336, 26881536,
                                  27956736, 29031936, 30107136, 31182336, 32257536, 33332736, 34407936, 35483136,
                                  36558336, 37633536, 38708736, 39783936, 40859136, 41934336, 43009536, 44084736,
                                  45159936, 46235136, 47310336, 48385536, 49460736, 49999872]
'''

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='pretrained_ppo', help='specifies whether frames should be collected using a random or pretrained agent')
    parser.add_argument('--save_dir', type=str, default=os.getcwd() + '/atari_images', help='directory to save images')
    parser.add_argument('--game', type=str, default='RiverraidNoFrameskip-v4', help='game from which to gather frames')
    parser.add_argument('--num_frames', type=int, default=1000, help='number of frames to gather (across multiple episodes)')
    parser.add_argument('--save_file_prefix', type=str, default=None, help='filename to prepend to save images')
    parser.add_argument('--frames_per_epoch', type=int, default=1., help='percentage of frames to gather from a single episode')
    parser.add_argument('--randomize_collection_start_point', action='store_true', help='flag indicating whether frames should be collected from start to \
        end of episode or collection should start at a random frame within the episode')
    parser.add_argument('--num_frame_stack', type=int, default=1, help='Number of frames to stack together')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of threads to use to collect frames')
    parser.add_argument('--color', type=bool, default=True, help='Flag indicating whether frames are color or grayscale')
    parser.add_argument('---downsample', type=bool, default=False, help='Flag indictaing whether to downsample frames')
    parser.add_argument('--checkpoint_index', type=int, default=-1, help='index of saved checkpoints to use for ppo agent')
    parser.add_argument('--min_episode_length', type=int, default=40, help='Min length of an episode')
    parser.add_argument('--starting_example_number',type=int, default=0, help='starting number to append to filename')
    return parser.parse_args()



def gather_frames():
    opt = parse_arguments()

    ## setup save directory ## 
    if not os.path.exists(opt.save_dir) or not os.path.isdir(opt.save_dir):
        try:
            os.mkdir(opt.save_dir)
        except OSError:
            print ("Creation of the directory %s failed" % opt.save_dir)
            return
        else:
            print ("Successfully created the directory %s " % opt.save_dir)
    
    ## set filename for images ##
    game_name = re.search("(?:(?!No).)*", opt.game).group(0) # Include everything up until No
    filename = game_name
    if opt.save_file_prefix:
        filename = opt.save_file_prefix
    
    i = opt.starting_example_number
    seed = 42
   
    '''
    while i < opt.num_frames:
        steps = 0
        while steps/opt.num_frames < opt.frames_per_epoch:
            steps += 1
    '''
    total_iter = 0
    while total_iter < opt.num_frames:
        if opt.agent_type == "random_agent":
            # List of episodes. Each episode is a list of 160x210 observations
            episodes, episode_labels = get_random_agent_rollouts(env_name=opt.game, steps=opt.num_frames, seed=seed, num_processes=opt.num_processes,
                                                             num_frame_stack=opt.num_frame_stack, downsample=opt.downsample, color=opt.color)

        elif opt.agent_type == "pretrained_ppo":
            # List of episodes. Each episode is a list of 160x210 observations
            episodes, episode_labels = get_ppo_rollouts(env_name=opt.game, steps=opt.num_frames, seed=seed, num_processes=opt.num_processes,
                                                             num_frame_stack=opt.num_frame_stack, downsample=opt.downsample, color=opt.color,
                                                             checkpoint_index=opt.checkpoint_index)

        else:
            assert False, "Collect mode {} not recognized".format(opt.agent_type)
    
        # Get indices for episodes that have min_episode_length
        ep_inds = [j for j in range(len(episodes)) if len(episodes[j]) > opt.min_episode_length]
        episodes = [episodes[j] for j in ep_inds]
        print("Number of episodes:", len(episodes))

        for episode in episodes:
            if total_iter >= opt.num_frames:
                break
            start = 0
            if opt.randomize_collection_start_point:
                start = np.random.choice(len(episode),1)[0]
                print("Collection start point", start)
            ep_steps = 0
            for _ in range(len(episode)):
                ep = episode[_].permute((1,2,0)).numpy()
                image = Image.fromarray(ep)
                image.save(os.path.join(opt.save_dir, filename + '_' + str(i) + '.png'))
                i += 1
                total_iter += 1
                ep_steps += 1
                if total_iter >= opt.num_frames:
                    break
            print("total episode steps:",ep_steps)
        
        




if __name__=="__main__":
    gather_frames()