import os
import json
import csv
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# models = genai.list_models()
# for m in models:
#     print(m.name, m.supported_generation_methods)

# Load JSON data from the file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_to_summary(data):
    has_locked_reason = False
    merged = False
    num_comments = 0
    num_review_comments = 0

    summary = (
        f"Pull Request '{data['number']}' titled '{data['title']}' was authored by a {data['user']['type']}, who is associated as a {data['author_association']}. "
        f"\nIt was created at {data['created_at']}, and was closed at {data['closed_at']} {('by a ' + data['closed_by']['type']) if data.get('closed_by') else 'N/A'}.\n"
    )

    if 'labels' in data:
        labels = data['labels']
        if len(labels) > 0:
            labels_text = 'The PR has labels: '
            for i in range(len(labels)):
                if i != len(labels)-1:
                    labels_text += f"{labels[i]['name']} - {labels[i]['description']}, "
                else:
                    labels_text += f"{labels[i]['name']} - {labels[i]['description']}. "
            summary += labels_text + '\n'

    if 'locked' in data and data["locked"] and 'active_lock_reason' in data and data['active_lock_reason']:
        has_locked_reason = True
        summary += f"PR was locked because of {data['active_lock_reason']}.\n"

    if data['body']:
        summary += f"It has a body of '{data['body']}'\n"

    if data['comments_url_body']:
        comments = data['comments_url_body']
        num_comments = len(comments)
        summary += f"PR has comments:\n"
        for i in range(num_comments):
            summary += f"'{comments[i]['body']}' by a {comments[i]['author_association']} of type {comments[i]['user']['type']} on {comments[i]['created_at']}\n"
        summary += '\n'

    if 'pull_request' in data:
        if data['pull_request']['merged_at']:
            merged = True
            summary += f"It was merged at {data['pull_request']['merged_at']} by a {data['pull_request_url_body']['merged_by']['type']}.\n "

        # if data['pull_request']['patch_url_body']:
        #     summary += f"The PR includes the following patch:\n{data['pull_request']['patch_url_body']}.\n"

        if data['pull_request_url_body']['review_comments_url_body']:
            review_comments = data['pull_request_url_body']['review_comments_url_body']
            num_review_comments = len(review_comments)
            summary += f"PR has review comments:\n"
            for i in range(num_review_comments):
                summary += f"'{review_comments[i]['body']}' by a {review_comments[i]['author_association']} of type {review_comments[i]['user']['type']} on {review_comments[i]['created_at']}\n"
            summary += '\n'
            
    return summary.strip(), has_locked_reason, merged, num_comments, num_review_comments

def generate_prompt(summary):
    return f"""
                Given this summary of PR closure: "{summary}", what is the reason for closure?

                Refer to these PR summary and the reason to generate a 10 words which gives a general reason like the ones mentioned:

                The Reason for this: "Pull Request '25802' titled 'better error message when lowering a primitive with a custom_partition rule' was authored by a User, who is associated as a NONE. \nIt was created at 2025-01-09T12:38:22Z, and was closed at 2025-01-09T15:56:08Z by a User.\nIt has a body of 'Hello,\r\n\r\nA quick PR to improve an error message I got when I was using `custom_partitionning`\r\n\r\nHere is a MWE\r\n\r\n```py\r\nimport os\r\n\r\nos.environ[\"JAX_PLATFORM_NAME\"] = \"cpu\"\r\nos.environ[\"XLA_FLAGS\"] = \"--xla_force_host_platform_device_count=8\"\r\n\r\nfrom jax import core, lax\r\nimport jax.numpy as jnp\r\nimport jax\r\nfrom jax.experimental.custom_partitioning import custom_partitioning\r\nfrom jax.interpreters import mlir\r\nimport jax.extend as jex\r\n\r\nfrom jax.sharding import NamedSharding\r\nfrom jax.sharding import PartitionSpec as P\r\n\r\npdims = (8,)\r\nmesh = jax.make_mesh(pdims, axis_names=(\"x\"))\r\nsharding = NamedSharding(mesh, P(\"x\"))\r\n\r\n# ================================\r\n# Double Primitive Rules\r\n# ================================\r\n\r\n# Step 1: Define the Primitive\r\ndouble_prim_p = jex.core.Primitive(\"double_prim\")\r\n\r\n\r\n# dispatch.prim_requires_devices_during_lowering.add(double_prim_p)\r\n# Step 2: Define the Implementation\r\n@custom_partitioning\r\ndef double_prim_impl(x):\r\n    return 2 * x  # Linear operation\r\n\r\n\r\ndef infer_sharding_from_operands(mesh, arg_infos, result_infos):\r\n    return arg_infos[0].sharding\r\n\r\n\r\ndef partition(mesh, arg_infos, result_infos):\r\n    input_sharding = arg_infos[0].sharding\r\n    output_sharding = result_infos.sharding\r\n    input_mesh = input_sharding.mesh\r\n\r\n    def impl(operand):\r\n        return 2 * operand\r\n\r\n    return input_mesh, impl, output_sharding, (input_sharding,)\r\n\r\n\r\n# Step 3: Define Abstract Evaluation\r\ndef double_prim_abstract_eval(x):\r\n    return core.ShapedArray(x.shape, x.dtype)\r\n\r\n\r\n# Step 4: Register the Primitive\r\ndouble_prim_p.def_impl(double_prim_impl)  # Implementation\r\ndouble_prim_p.def_abstract_eval(double_prim_abstract_eval)  # Abstract Eval\r\ndouble_prim_impl.def_partition(\r\n    infer_sharding_from_operands=infer_sharding_from_operands, partition=partition\r\n)\r\nmlir.register_lowering(\r\n    double_prim_p, mlir.lower_fun(double_prim_impl, multiple_results=False)\r\n)  # Lowering\r\n\r\n\r\n# Define a Python wrapper for the primitive\r\n@jax.jit\r\ndef double_prim_call(x):\r\n    return double_prim_p.bind(x)\r\n\r\n# Test Forward Computation\r\nx = jnp.arange(8).astype(jnp.float32)\r\nx = lax.with_sharding_constraint(x, sharding)\r\nprint(\"Double Primitive Forward:\\n\", double_prim_call(x))\r\n```\r\n\r\nI get this error message \r\n`Please file a bug at https://github.com/jax-ml/jax/issues`\r\n\r\nIf I uncomment this line \r\nit is fixed\r\n\r\n`dispatch.prim_requires_devices_during_lowering.add(double_prim_p)`\r\n\r\nI added a more explicit error message.'\nPR has comments:\n'Thanks for your pull request! It looks like this may be your first contribution to a Google open source project. Before we can look at your pull request, you'll need to sign a Contributor License Agreement (CLA).\n\nView this [failed invocation](https://github.com/jax-ml/jax/pull/25802/checks?check_run_id=35370052235) of the CLA check for more information.\n\nFor the most up to date status, view the checks section at the bottom of the pull request.' by a NONE of type Bot on 2025-01-09T12:38:26Z\n'Thanks for this! But, I don't think that this is quite the right change. This new error message would expose JAX internal APIs which we definitely don't want to do. I would argue that this is actually really a misuse of `custom_partitioning`, so I'd say that the error message is actually perfect :D \r\n\r\nHere's how I would recommend refactoring the code. Instead of wrapping your primitive impl in `custom_partitioning`, move that to `double_prim_call`:\r\n\r\n```diff\r\n...\r\n\r\n# Step 2: Define the Implementation\r\n- @custom_partitioning\r\ndef double_prim_impl(x):\r\n    return 2 * x  # Linear operation\r\n\r\n...\r\n\r\n+ @custom_partitioning\r\ndef double_prim_call(x):\r\n    return double_prim_p.bind(x)\r\n\r\n...\r\n\r\n+ double_prim_call.def_partition(\r\n+     infer_sharding_from_operands=infer_sharding_from_operands, partition=partition\r\n+ )\r\n```\r\n\r\nAnd then everything works as expected!\r\n\r\nHope this helps.' by a COLLABORATOR of type User on 2025-01-09T13:35:49Z\n'Thank you for your answer\r\n\r\nThe problem is custom_partition lowering is not vmappable nor differentiable\r\nthis is an example of what you suggest\r\n\r\n```py\r\nimport os\r\nos.environ[\"JAX_PLATFORM_NAME\"] = \"cpu\"\r\nos.environ[\"XLA_FLAGS\"] = \"--xla_force_host_platform_device_count=8\"\r\n\r\nfrom jax import core, lax\r\nfrom jax.interpreters import mlir, ad, batching\r\nimport jax.numpy as jnp\r\nimport jax\r\nfrom jax.experimental.custom_partitioning import custom_partitioning\r\nfrom jax._src import dispatch\r\nimport jax.extend as jex\r\n\r\nfrom jax.sharding import NamedSharding\r\nfrom jax.sharding import PartitionSpec as P\r\nfrom functools import partial\r\n\r\npdims = (8,)\r\nmesh = jax.make_mesh(pdims , axis_names=('x'))\r\nsharding = NamedSharding(mesh, P('x'))\r\n\r\n# ================================\r\n# Double Primitive Rules\r\n# ================================\r\n\r\n# Step 1: Define the Primitive\r\ndouble_prim_p = jex.core.Primitive(\"double_prim\")\r\ndispatch.prim_requires_devices_during_lowering.add(double_prim_p)\r\n# Step 2: Define the Implementation\r\n#@custom_partitioning\r\ndef double_prim_impl(x):\r\n    return 2 * x  # Linear operation\r\n\r\ndef infer_sharding_from_operands(mesh , arg_infos , result_infos):\r\n    return arg_infos[0].sharding\r\n\r\ndef partition(mesh , arg_infos , result_infos):\r\n\r\n    input_sharding = arg_infos[0].sharding\r\n    output_sharding = result_infos.sharding\r\n    input_mesh = input_sharding.mesh\r\n\r\n    def impl(operand):\r\n        return 2 * operand\r\n\r\n    return input_mesh , impl , output_sharding , (input_sharding,)    \r\n\r\n@partial(custom_partitioning , static_argnums=(1,))\r\ndef vmapped_double_prim_impl(x, batch_dims):\r\n    return jax.vmap(lambda x: 2 * x , in_axes=batch_dims)(x)\r\n\r\ndef v_infer_sharding_from_operands(batch_dims , mesh , arg_infos , result_infos):\r\n    return arg_infos[0].sharding\r\n\r\ndef v_partition(batch_dims , mesh , arg_infos , result_infos):\r\n    input_sharding = arg_infos[0].sharding\r\n    output_sharding = result_infos.sharding\r\n    input_mesh = input_sharding.mesh\r\n\r\n    def impl(operand):\r\n        return jax.vmap(lambda x: 2 * x , in_axes=batch_dims)(operand)\r\n\r\n    return input_mesh , impl , output_sharding , (input_sharding,)\r\n\r\n\r\nvmapped_double_prim_impl.def_partition(infer_sharding_from_operands=v_infer_sharding_from_operands, partition=v_partition)\r\n\r\n# Step 3: Define Abstract Evaluation\r\ndef double_prim_abstract_eval(x):\r\n    return core.ShapedArray(x.shape, x.dtype)\r\n\r\n# Step 4: Define JVP Rule\r\ndef double_prim_jvp_rule(primals, tangents):\r\n    x, = primals\r\n    t, = tangents\r\n\r\n    # Forward computation\r\n    primal_out = double_prim_call(x)\r\n\r\n    # Tangent computation (reuse the primitive itself)\r\n    tangent_out = double_prim_call(t)\r\n    return primal_out, tangent_out\r\n\r\n# Step 5: Define Transpose Rule\r\ndef double_prim_transpose_rule(ct_out, x):\r\n    ct_x = 2*ct_out if ad.is_undefined_primal(x) else None\r\n    return ct_x ,\r\n\r\n# Step 6: Define Batch Rule\r\ndef double_prim_batch_rule(batched_args, batch_dims):\r\n    x, = batched_args\r\n    bx, = batch_dims\r\n    # Apply vmapped double operation\r\n    res = vmapped_double_prim_impl(x, bx)\r\n    return res, 0\r\n\r\n# Step 7: Register the Primitive\r\ndouble_prim_p.def_impl(double_prim_impl)  # Implementation\r\ndouble_prim_p.def_abstract_eval(double_prim_abstract_eval)  # Abstract Eval\r\nmlir.register_lowering(double_prim_p, mlir.lower_fun(double_prim_impl, multiple_results=False))  # Lowering\r\nad.primitive_jvps[double_prim_p] = double_prim_jvp_rule  # JVP Rule\r\nad.primitive_transposes[double_prim_p] = double_prim_transpose_rule  # Transpose Rule\r\nbatching.primitive_batchers[double_prim_p] = double_prim_batch_rule  # Batch Rule\r\n\r\n\r\n# Define a Python wrapper for the primitive\r\n@custom_partitioning\r\ndef double_prim_call(x):\r\n    return double_prim_p.bind(x)\r\n\r\ndouble_prim_call.def_partition(infer_sharding_from_operands=infer_sharding_from_operands, partition=partition)\r\n\r\n# ================================\r\n# Linear Double Primitive Testing\r\n# ================================\r\n\r\n# Test Forward Computation\r\nx = jnp.arange(8).astype(jnp.float32)\r\nx = lax.with_sharding_constraint(x, sharding)\r\nprint(\"Double Primitive Forward:\\n\", double_prim_call(x))\r\n\r\n# Test Reverse-Mode Autodiff\r\nprint(\"Double Primitive Grad:\\n\", jax.jacrev(double_prim_call)(x))\r\nprint(\"Double Primitive VJP:\\n\", jax.vjp(double_prim_call, x)[0])\r\n\r\n## Test Forward-Mode Autodiff \r\n# print(f\"Double Primitive Forward diff:\\n\", jax.jacfwd(double_prim_call)(x)) THIS IS CRASHING IN ALL CASES BECAUSE THE SHARDING IS NOT PROPAGATED CORRECTLY IN JACFWD\r\nprint(\"Double Primitive jvp:\\n\", jax.jvp(double_prim_call, (x,),( jnp.ones_like(x),)))\r\n\r\n# Test Batch Rule\r\nbatched_x = jnp.stack([x, 2* x])\r\nprint(\"Double Primitive Batched:\\n\",jax.vmap(double_prim_call, in_axes=0)(batched_x))\r\n```\r\n\r\nYou can see that there is no longer any differentiation rules nor batch rules when wrapping with `custom_partitionning`\r\n\r\nThe whole goal of what I am doing is implementing the batching and diff rules \r\n\r\nA working example is then\r\n```py\r\n@custom_partitioning\r\ndef double_prim_impl(x):\r\n    return 2 * x  # Linear operation\r\n    \r\n@jax.jit\r\ndef double_prim_call(x):\r\n    return double_prim_p.bind(x)\r\n\r\ndouble_prim_impl.def_partition(infer_sharding_from_operands=infer_sharding_from_operands, partition=partition)\r\n```\r\nThe custom_partitinniong needs to wrapped with a primitive that defines all rules\r\n\r\nThere is still no way to do a jacfwd .. the vmap however works the way I did it\r\n\r\nPlease tell me if I am going the wrong way' by a NONE of type User on 2025-01-09T13:51:47Z\n\nPR has review comments:\n'I don't think we want to put private APIs in the error message.\r\n\r\nNow I know that's suboptimal but maybe we can document it in some other way in the docs somewhere' by a COLLABORATOR of type User on 2025-01-09T15:42:58Z\n'It was mentioned in the changelog: https://github.com/jax-ml/jax/blob/main/CHANGELOG.md#jax-0424-feb-6-2024\r\n\r\nmaybe that's enough for now?' by a COLLABORATOR of type User on 2025-01-09T15:47:02Z\n'Ok good for me\r\nI will close this PR' by a NONE of type User on 2025-01-09T15:56:03Z" is "Erroneous :- PR does not add value, rather risks exposing internal APIs."

                The Reason for this: "Pull Request '25143' titled 'Disallow platform aliases for get_topology_desc' was authored by a User, who is associated as a CONTRIBUTOR. \nIt was created at 2024-11-27T08:54:52Z, and was closed at 2024-12-12T07:52:29Z by a User.\nThe PR has labels: pull ready - Ready for copybara import and testing. \nPR has comments:\n'What is the status? It seem approved, but not merged.' by a COLLABORATOR of type User on 2024-12-04T15:39:30Z\n'Abandoning (since deviceless AOT is already working without this).' by a CONTRIBUTOR of type User on 2024-12-12T07:52:29Z\n\nPR has review comments:\n'`if platform not in expand_platform_alias(platform):`. Avoid using `if not ... in ...` pattern.' by a COLLABORATOR of type User on 2024-11-27T16:42:17Z\n'Done.' by a CONTRIBUTOR of type User on 2024-11-27T18:25:52Z" is "Redundant :- issue already working without the PR."

                The Reason for this: "Pull Request '25107' titled 'Track mapping of platform aliases to compile-only backends' was authored by a User, who is associated as a CONTRIBUTOR. \nIt was created at 2024-11-26T08:48:39Z, and was closed at 2024-12-03T07:58:16Z by a User.\nThe PR has labels: pull ready - Ready for copybara import and testing. \nIt has a body of 'The patch tracks the mapping of aliases to compile-only backend platform names. The mapping enables canonicalizing platform names correctly ('gpu' -> 'cuda') when we only have compile-only backends for the platform.'\nPR has comments:\n'This is an alternative to  https://github.com/jax-ml/jax/pull/25033 to address https://github.com/jax-ml/jax/issues/23971.' by a CONTRIBUTOR of type User on 2024-11-26T08:53:54Z\n\nPR has review comments:\n'Do not update config like this without a context manager. Can you set the config just for this test or in setUp and revert it back in tearDown?' by a COLLABORATOR of type User on 2024-11-27T16:46:35Z\n'The function runs in a different process, so it should not matter, no?' by a CONTRIBUTOR of type User on 2024-11-27T18:19:05Z\n'No, pytest can run stuff in the same process which can affect other tests too. Also, it's a good practice in general to scope global updates to what you need.' by a COLLABORATOR of type User on 2024-11-27T18:28:50Z\n'WHy global_config_context? Looks like this should be local?' by a COLLABORATOR of type User on 2024-11-28T16:15:42Z\n'I thought you asked me to use a context manager that will remember the original state of the config and restore it when the test is done. This seems to be exactly what global_config_context is doing: https://github.com/jax-ml/jax/blob/aff7714dc0f49cc0097e4db08e028b68182c8ab9/jax/_src/test_util.py#L1167 (I assumed the name implies that it is temporarily updating the *global* JAX config). Is there some local version of this context manager?' by a CONTRIBUTOR of type User on 2024-12-02T16:53:51Z" is "Miscommunication :- author did not understand the changes reviewer wanted." 

                The Reason for this: "Pull Request '24438' titled 'Alternative abs() formula for `sph_harm()` with certain GPU/CUDA combinations' was authored by a User, who is associated as a COLLABORATOR. \nIt was created at 2024-10-21T21:04:26Z, and was closed at 2024-11-15T05:25:39Z by a User.\nIt has a body of 'It works around a known ptxas optimization bug, which causes abs() inside array indices to be lost and leads to incorrect clamping of negative indices at 0. The bug causes the `jax.scipy.special.sph_harm` function to produce incorrect results on CUDA GPUs of compute capability 9.0. This issue only affects CUDA Toolkit versions 12.5.0 to 12.6.2 due to a known compiler bug, which has been resolved in subsequent releases.\r\n\r\nMinimal reproducer for the bug (on a C.C. 9.0 device such as H100):\r\n```\r\n$ docker run -it --gpus all --shm-size=1g ghcr.io/nvidia/jax:jax-2024-10-20 bash\r\n# python <<EOF\r\nimport jax\r\nimport jax.numpy as jnp\r\nfrom jax.scipy.special import sph_harm\r\nfrom scipy import special\r\nm = jnp.arange(-3, 3)[:, None]\r\nn = jnp.arange(3, 6)\r\nn_max = 5\r\ntheta = 0.0\r\nphi = jnp.pi\r\nprint(sph_harm(m, n, theta, phi, n_max=n_max))\r\nprint(special.sph_harm(m, n, theta, phi))\r\nEOF\r\n```\r\nexample output:\r\n```\r\n[[ 0.7463527 -0.j -0.84628445+0.j  0.9356027 -0.j]\r\n [-0.7463527 +0.j  0.84628445+0.j -0.9356027 +0.j]\r\n [ 0.7463527 -0.j -0.84628445+0.j  0.9356027 -0.j]\r\n [-0.7463527 -0.j  0.84628445+0.j -0.9356027 -0.j]\r\n [ 0.        +0.j  0.        +0.j  0.        +0.j]\r\n [-0.        -0.j  0.        +0.j  0.        +0.j]]\r\n[[ 0.        -0.j -0.        +0.j -0.        +0.j]\r\n [ 0.        +0.j  0.        +0.j  0.        +0.j]\r\n [-0.        +0.j  0.        -0.j -0.        +0.j]\r\n [-0.74635267+0.j  0.84628438+0.j -0.93560258+0.j]\r\n [ 0.        +0.j -0.        +0.j  0.        +0.j]\r\n [-0.        +0.j  0.        +0.j  0.        +0.j]]\r\n```\r\n\r\n\r\nA distilled the repro which shows that the abs-indices gets incorrectly clamped:\r\n```\r\nimport jax\r\nimport jax.numpy as jnp\r\n\r\nA = jnp.arange(10).reshape(5, 2)\r\ni = jnp.arange(-2, 3)\r\nj = jnp.arange(2)\r\nprint(jax.jit(lambda A, i, j: A.at[jnp.abs(i)[:, None], j[None, :]].get(mode='clip'))(A, i, j))\r\n```'\nPR has comments:\n'Closing it as the underlying ptxas bug will be fixed via CUDA 12.6 U3.' by a COLLABORATOR of type User on 2024-11-15T05:25:39Z\n\nPR has review comments:\n'Note this is checking the wrong version. You should be checking the version of `ptxas`, not the version of libcudart. They may be the same, they may not.' by a COLLABORATOR of type User on 2024-10-25T13:36:54Z\n'Is there a jax/jaxlib API somewhere that exposes the ptxas version? If not, what is the recommended way to locate the ptxas binary that XLA uses (assuming if multiple versions of ptxas may co-exist on a system)?' by a COLLABORATOR of type User on 2024-10-28T03:58:25Z" is "Outdated :- PR worked around a bug which was fixed by the dependency version upgrade."

                The Reason for this: "Pull Request '24203' titled 'Drop `complex dtype` support in `jnp.arctan2` to make it consistent with `np.arctan2`' was authored by a User, who is associated as a CONTRIBUTOR. \nIt was created at 2024-10-09T08:03:08Z, and was closed at 2024-10-17T02:14:58Z by a User.\nIt has a body of 'This PR fixes a discrepancy between `jnp.arctan2` and `np.arctan2` by raising a `TypeError` for `complex` inputs, as `np.arctan2` currently does.\r\n\r\nCurrent behavior:\r\n```python\r\n>>> jnp.arctan2(1-2j, 3)\r\nArray(0.4913969-0.6412373j, dtype=complex64, weak_type=True)\r\n```\r\n\r\nNew behavior:\r\n```python\r\n>>> jnp.arctan2(1-2j, 3)\r\nTypeError: ufunc 'arctan2/atan2' does not support complex dtypes.\r\n```'\nPR has comments:\n'Does the function return reasonable results for complex input? If so, I don't think we should do this deprecation, as it would potentially break existing users with very little benefit.' by a COLLABORATOR of type User on 2024-10-09T12:21:18Z\n'@pearu would probably have the best opinion on \"does this function return reasonable results for complex inputs?\".' by a COLLABORATOR of type User on 2024-10-09T14:13:56Z\n'It dispatches to `lax.atan2`, so I suspect the answer is yes for the sake of this discussion. (it's not returning garbage, it's actually attempting a valid computation).\r\n\r\nMy thinking here: in general JAX functionality is a superset of numpy functionality. So just because NumPy returns a TypeError doesn't mean JAX must as well. Does that make sense?' by a COLLABORATOR of type User on 2024-10-09T14:29:37Z\n'Thanks for the clarification @jakevdp. Can I modify this PR to add the docstring for `arctan2`?' by a CONTRIBUTOR of type User on 2024-10-10T05:08:28Z\n'atan2 is implemented in stablehlo, see https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2 . Usually, arctan2 for complex inputs is not supported (for example, by numpy, torch, Python array API standard v2023.12, C++ numerics library, etc) as `atan2(y, x)` is associated with the direction angle of the point `(x, y)` in the  Cartesian coordinates. However, numerically, supporting `atan2` for complex inputs does make sense for cases where `x` is complex zero or close to complex zero, and ideally, `atan2(y, x)` should be more accurate than `atan(y / x)`.\r\n\r\nThat said, the current implementation of atan2 on complex inputs in stablehlo is problematic accuracy-wise. For example:\r\n```python\r\n>> x, y = 1+0.00001j, 1-0.00001j\r\n>>> jnp.arctan2(y, x)   # the imaginary part is inaccurate\r\nArray(0.7853981-1.001353e-05j, dtype=complex64, weak_type=True)\r\n>>> jnp.arctan(y / x)   # expected\r\nArray(0.7853982-1.e-05j, dtype=complex64, weak_type=True)\r\n```\r\nand atm I would recommend using `atan(y / x)` instead of `atan2(y, x)` when `x != 0+0j` (I'll add this issue to my todo list).\r\n\r\nOn the other hand, to allow the switch `numpy <-> jax.numpy` in both directions, the current PR makes sense, although, I second @jakevdp point that jax.numpy will likely never be equivalent to `numpy`.' by a COLLABORATOR of type User on 2024-10-11T10:20:42Z\n'I think we should close this PR, because we shouldn't deprecate or raise an error for complex inputs to this function.' by a COLLABORATOR of type User on 2024-10-16T16:13:26Z\n'Thanks! Closing the PR.' by a CONTRIBUTOR of type User on 2024-10-17T02:15:36Z\n\nPR has review comments:\n'Raising an exception does not correspond to deprecation which I would expect to trigger a warning for a few releases. So, I suggest fixing the title of the PR: `Deprecate` -> `Drop` or similar.' by a COLLABORATOR of type User on 2024-10-12T09:43:30Z\n'Thanks! Modified the title.' by a CONTRIBUTOR of type User on 2024-10-16T06:35:55Z" is "Erroneous :- PR does not add value, rather deprecates current implementation."
            """

def call_gemini(prompts):
    results = []
    batch_size = 10

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        try:
            responses = [model.generate_content(p) for p in batch]
            reasons = [resp.text.strip() if resp.text else "No response" for resp in responses]
        except Exception as e:
            print(f"Error during API call: {e}")
            reasons = ["API Error"] * len(batch)

        results.extend(reasons)
        sleep(2)  # Prevent hitting rate limits

    return results

def main():
    # input_file = os.path.join(os.path.dirname(__file__), "../example_data/cleaned_extended_issue.json")
    input_file = os.path.join(os.path.dirname(__file__), "../scraped_data/jax-ml_jax.json")
    output_json = os.path.join(os.path.dirname(__file__), "pr_closure_reasons.json")
    output_json_just_reasons = os.path.join(os.path.dirname(__file__), "pr_closure_just_reasons.json")

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    data = load_json(input_file)
    total = 0
    count = 0
    if isinstance(data, list):  # If JSON is a list of PRs, process each one
        summary_data = []
        for entry in data:
            total += 1
            num_comments = len(entry['comments_url_body'])
            if 'pull_request' in entry and entry['pull_request']:
                num_review_comments = len(entry['pull_request_url_body']['review_comments_url_body'])
                pull_request = entry['pull_request']
                if not pull_request.get('merged_at') and num_comments > 0 and num_review_comments > 0:
                    count += 1
                    summary_data.append(json_to_summary(entry))

    else:  # If JSON contains only a single PR, process it directly
        if 'pull_request' in data and data['pull_request'] and not data['pull_request'].get('merged_at'):
            summary_data = [json_to_summary(data)]
        else:
            summary_data = []  # or handle the case where no matching PR is found

    summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list = zip(*summary_data)

    print(f"Generating reasons for: {count} out of {total}")

    prompts = [generate_prompt(summary) for summary in summarys]

    print("Sending requests to Gemini API...")
    reasons = call_gemini(prompts)
    
    # Prepare the result data in a JSON-compatible format
    results = [
        {
            "summary": summary,
            "has_locked_reason": has_locked_reason,
            "merged": merged,
            "num_comments": num_comments,
            "num_review_comments": num_review_comments,
            "reason_for_closure": reasons
        }
        for summary, has_locked_reason, merged, num_comments, num_review_comments, reasons in zip(summarys, has_locked_reasons_list, is_merged_list, num_comments_list, num_review_comments_list, reasons)
    ]


    # Save results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json}")

    with open(output_json_just_reasons, "w", encoding="utf-8") as f:
        json.dump(reasons, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json_just_reasons}")

if __name__ == "__main__":
    main()