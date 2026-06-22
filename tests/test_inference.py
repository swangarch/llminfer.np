from inference_acceleration import main


def test_without_acceleration():
    assert main([]) == """The following is a conversation between a User and a helpful Assistant.

    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant: The capital of France is Paris.


User: I'm using it to register a new account. Do you want to invite me to your account?

Assistant: You don't have to invite me to the account, but you do have to do something that you normally would not do with someone else. I'll be at the account.


User: Is your account private?

Assistant: If you're using my name and password, or your"""


def test_KV_cache():
    assert main(["-kv"]) == """The following is a conversation between a User and a helpful Assistant.

    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant: The capital of France is Paris.


User: I'm using it to register a new account. Do you want to invite me to your account?

Assistant: You don't have to invite me to the account, but you do have to do something that you normally would not do with someone else. I'll be at the account.


User: Is your account private?

Assistant: If you're using my name and password, or your"""