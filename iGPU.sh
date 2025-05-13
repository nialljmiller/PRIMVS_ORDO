#!/bin/bash

# Default time for interactive session (minimum 8 hours)
DEFAULT_TIME="8:00:00"

# Default account and QoS (adjust according to your use case)
DEFAULT_ACCOUNT="joycelab-niall"
DEFAULT_QOS="interactive"

# Function to request compute resources (Low Tier - A30)
request_low() {
	    echo "Requesting Low Resource Tier: 1x A30 GPU (24GB VRAM), 64GB RAM, 8 hours"
	        salloc --job-name=contrastive_gpu \
			           --time=$DEFAULT_TIME \
				              --gres=gpu:1 \
					                 --mem=64G \
							            --partition=mb-a30 \
								               --account=$DEFAULT_ACCOUNT \
									                  --qos=$DEFAULT_QOS
										  }

									  # Function to request compute resources (Medium Tier - L40S)
									  request_medium() {
										      echo "Requesting Medium Resource Tier: 1x L40S GPU (48GB VRAM), 128GB RAM, 8 hours"
										          salloc --job-name=contrastive_gpu \
												             --time=$DEFAULT_TIME \
													                --gres=gpu:1 \
															           --mem=128G \
																              --partition=mb-l40s \
																	                 --account=$DEFAULT_ACCOUNT \
																			            --qos=$DEFAULT_QOS
																			    }

																		    # Function to request compute resources (High Tier - H100)
																		    request_high() {
																			        echo "Requesting High Resource Tier: 1x H100 GPU (80GB VRAM), 256GB RAM, 8 hours"
																				    salloc --job-name=contrastive_gpu \
																					               --time=$DEFAULT_TIME \
																						                  --gres=gpu:1 \
																								             --mem=256G \
																									                --partition=mb-h100 \
																											           --account=$DEFAULT_ACCOUNT \
																												              --qos=$DEFAULT_QOS
																												      }

																											      # Usage message
																											      usage() {
																												          echo "Usage: $0 {low|medium|high}"
																													      echo "low    - Request low-tier resources (1x A30 GPU, 64GB RAM, 8 hours)"
																													          echo "medium - Request medium-tier resources (1x L40S GPU, 128GB RAM, 8 hours)"
																														      echo "high   - Request high-tier resources (1x H100 GPU, 256GB RAM, 8 hours)"
																													      }

																												      # Main logic to choose tier
																												      if [ $# -ne 1 ]; then
																													          usage
																														      exit 1
																												      fi

																												      case $1 in
																													          low)
																															          request_low
																																          ;;
																																	      medium)
																																		              request_medium
																																			              ;;
																																				          high)
																																						          request_high
																																							          ;;
																																								      *)
																																									              usage
																																										              exit 1
																																											              ;;
																																										      esac

