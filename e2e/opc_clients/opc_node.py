from asyncua import Node
import logging
import random

logger = logging.getLogger(__name__)


class ServerNode:
    def __init__(
            self, tag_name: str, namespace: int, initial_val: float, root_node: Node, dropout: bool = True
    ) -> None:
        self.namespace = namespace
        self.tag_name = tag_name
        self.initial_val = initial_val
        self.root_node = root_node
        self.good_status = True
        self.bad_counter = 0
        self.node: Node | None = None
        self.dropout = dropout

    async def initialize_node(self) -> None:
        nodeid = self.get_nodeid(ns=self.namespace, tag_name=self.tag_name)
        logger.warning(f"Creating node {nodeid}")
        node = await self.root_node.add_variable(
            nodeid, self.tag_name, self.initial_val
        )
        await node.set_writable()
        self.node = node

    async def step(self) -> None:
        """
        Increments bad counter if status is bad.
        If bad counter exceeds 30, flips status to good
        """

        if self.good_status:
            go_bad = random.random() < 0.001 and self.dropout
            if go_bad:
                await self.fail()
            return

        # status is bad
        self.bad_counter += 1
        if self.bad_counter >= 30:
            await self.recover()

    async def fail(self) -> None:
        assert self.node is not None
        self.good_status = False
        try:
            await self.node.delete()
            self.node = None
            logger.warning(f"Deleted node associated with tag {self.tag_name}")
        except Exception:
            logger.warning(f"Failed to delete node associated with tag {self.tag_name}")

    async def recover(self) -> None:
        # recreate deleted tag
        try:
            await self.initialize_node()
            self.good_status = True
            self.bad_counter = 0
        except Exception:
            logger.warning(
                f"Failed to create node associated with tag: {self.tag_name}"
            )

    def get_nodeid(self, ns: int, tag_name: str) -> str:
        if ";" in tag_name:
            # tag name is already formatted
            # swap from i to s indentifier
            # tag_name = tag_name.replace("i=", "s=")
            return tag_name
        # tag name is not yet formatted
        return f"ns={ns};s={tag_name}"
