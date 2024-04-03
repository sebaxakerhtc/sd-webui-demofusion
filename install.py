import launch

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers~=0.21.4", "requirements for DemoFusion")
#
# if not launch.is_installed("torch"):
#     launch.run_pip("install torch~=2.1.0", "requirements for DemoFusion")
#
# if not launch.is_installed("scipy"):
#     launch.run_pip("install scipy~=1.11.3", "requirements for DemoFusion")
#
# if not launch.is_installed("omegaconf"):
#     launch.run_pip("install omegaconf~=2.3.0", "requirements for DemoFusion")
#
# if not launch.is_installed("transformers"):
#     launch.run_pip("install transformers~=4.34.0", "requirements for DemoFusion")
#
# if not launch.is_installed("tqdm"):
#     launch.run_pip("install tqdm", "requirements for DemoFusion")
#
# if not launch.is_installed("matplotlib"):
#     launch.run_pip("install matplotlib", "requirements for DemoFusion")
