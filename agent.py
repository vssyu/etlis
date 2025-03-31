import pandas as pd
import duckdb
import re
from openai import OpenAI
import json
import sys
from io import StringIO

client = OpenAI(api_key="sk-5a3f1a8247324838a586ecb24b119373", base_url="https://api.deepseek.com")


def extract_python_code(text):
    # 定义正则表达式，匹配以```python开头和以```结尾之间的内容
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


class ChainOfThoughtAgent:
    @staticmethod
    def get_basic_attribution(df):
        col_len = len(df.columns)
        length = len(df)
        lis_col = list(df.columns)
        return col_len, length, lis_col

    def parse_response(self, text):
        tex_dict = json.loads(text)
        ans = []
        for k, v in tex_dict.items():
            ans.append(v)
        return ans

    def generate_steps(self, input_df, df_des, question):
        col_len, length, lis_col = ChainOfThoughtAgent.get_basic_attribution(input_df)
        prompt = """
## 您是资深数据分析师，需要将复杂问题拆解为原子化的分析步骤。
## 数据特征
数据集描述：$$$df_des$$$
行列规模：$$$length$$$行 × $$$col_len$$$列
列名：$$$lis_col$$$
主表名称：input_table

## 用户问题
$$$question$$$

## 任务要求
分析其中可能的原因，对表格做处理，写出分析需求，这个需求要指导代码生成，描述尽量详细，生成代码最好是可以返回一个根据需求计算出来的数字.或者和假设是否一致，是返回True或False，一个字典里面有你要求计算的字段和对应的值。你用的表头字段我已经全部给你了，没有其他字段。输出格式如下

{
"1":"xxx"
"2":"xxx"
}
不要输出其他东西
"""
        prompt = prompt.replace("$$$df_des$$$", str(df_des))
        prompt = prompt.replace("$$$length$$$", str(length))
        prompt = prompt.replace("$$$lis_col$$$", str(lis_col))
        prompt = prompt.replace("$$$col_len$$$", str(col_len))
        prompt = prompt.replace("$$$question$$$", str(question))
        print("ChainOfThoughtAgent_prompt", prompt)
        response = client.chat.completions.create(
            # model="deepseek-chat",
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.9,
            max_tokens=5000
        )

        return self.parse_response(response.choices[0].message.content)


class CodeExecutorAgent:

    @staticmethod
    def extract_python_code(text):
        # 定义正则表达式，匹配以```python开头和以```结尾之间的内容
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0]

    def get_code(self, text, df_des, input_df):
        col_len, length, lis_col = ChainOfThoughtAgent.get_basic_attribution(input_df)
        prompt = """
        我给你一个表叫input_table，这个表格是一个自动化率统计的表格。其中有2024年boq自动化占比，2024年参与人数，2024年boq数量，2025年boq自动化，2025年参与人数，2025年boq数量。boq自动化同比变化。boq自动化是一个百分比。参与人数人数代表的是处理人数boq的人数，boq数量是总的数量
数据集描述：$$$df_des$$$
行列规模：$$$length$$$行 × $$$col_len$$$列
列名：$$$lis_col$$$
主表名称：input_table

这里有一个需求：

$$$text$$$

你要写一个函数叫validate，输入是input_table，写上注释，只用写代码，其他不要输出。
"""
        prompt = prompt.replace("$$$df_des$$$", str(df_des))
        prompt = prompt.replace("$$$length$$$", str(length))
        prompt = prompt.replace("$$$lis_col$$$", str(lis_col))
        prompt = prompt.replace("$$$col_len$$$", str(col_len))
        prompt = prompt.replace("$$$text$$$", str(text))

        response = client.chat.completions.create(
            model="deepseek-reasoner",  # 使用实际模型名称
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=0.9,  # 添加DeepSeek特有参数
            max_tokens=2000  # 设置最大输出长度
        )
        return CodeExecutorAgent.extract_python_code(response.choices[0].message.content)

    @staticmethod
    def load_and_call_function(func_string, func_name, *args, **kwargs):
        """
        动态加载字符串形式的函数定义，并调用该函数。

        参数:
            func_string (str): 包含函数定义的字符串。
            func_name (str): 函数名。
            *args: 调用函数时的位置参数。
            **kwargs: 调用函数时的关键字参数。

        返回:
            dict: 包含执行结果和错误信息的字典。
        """
        # 创建一个局部命名空间
        local_namespace = {}

        try:
            # 使用 exec() 执行函数定义
            exec(func_string, {}, local_namespace)

            # 从命名空间中获取函数对象
            if func_name not in local_namespace:
                raise NameError(f"函数 '{func_name}' 未定义")

            func = local_namespace[func_name]

            # 调用函数并捕获返回值
            result_value = func(*args, **kwargs)

            return {
                "success": True,
                "result": result_value,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def run(self, text, input_df, df_des, max_retries=3):
        code = self.get_code(text, df_des, input_df)
        print(code)
        for attempt in range(max_retries):
            try:
                result = self.load_and_call_function(code, "validate", input_df)
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                code = self._correct_sql(code, str(e))
        raise Exception(f"代码执行失败")

    def _correct_code(self, bad_code, error):
        prompt = f"""修正以下DuckDB SQL错误：
错误信息：{error}
错误代码：{bad_code}
只需返回修正后的code，不要其他内容。"""

        response = client.chat.completions.create(
            model="deepseek-reasoner",  # 使用实际模型名称
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=0.9,  # 添加DeepSeek特有参数
            max_tokens=2000  # 设置最大输出长度
        )
        return CodeExecutorAgent.extract_python_code(response.choices[0].message.content.strip())


class SummaryAgent:
    def summarize(self, df_des, input_df, question, question_pair):
        col_len, length, lis_col = ChainOfThoughtAgent.get_basic_attribution(input_df)
        prompt = """
## 数据特征
数据集描述：$$$df_des$$$
行列规模：$$$length$$$行 × $$$col_len$$$列
列名：$$$lis_col$$$
主表名称：input_table
## 用户问题
$$$question$$$

##下面是可能的原因以及最后的验证结果
$$$new_question_pair$$$

你根据问题和可能的原因以及最后的验证结果总结一下,回答用户问题。
"""
        new_question_pair = []
        for k, v in question_pair.items():
            if v is not None:
                pp = k + "最后的结果是:" + str(v)
                new_question_pair.append(pp)
        new_question_pair = "\n\n".join(new_question_pair)
        prompt = prompt.replace("$$$df_des$$$", str(df_des))
        prompt = prompt.replace("$$$length$$$", str(length))
        prompt = prompt.replace("$$$lis_col$$$", str(lis_col))
        prompt = prompt.replace("$$$col_len$$$", str(col_len))
        prompt = prompt.replace("$$$question$$$", str(question))
        prompt = prompt.replace("$$$new_question_pair$$$", str(new_question_pair))
        print("SummaryAgent",prompt)
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # 使用实际模型名称
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.9,  # 添加DeepSeek特有参数
            max_tokens=2000  # 设置最大输出长度
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # 示例输入数据
    df = pd.read_excel(r"D:\download\mssc_data.xlsx")
    df_des = """这个表格提供了一个关于mssc组织在各个代表处的承接项目人数变化趋势的表格。每一行是一个项目，其中有，代表处，2024年MSSC承接人数，2024年代表处承接人数，2024MSSC承接人数占比，2025年MSSC承接人数，2025年代表处承接人数	，2025MSSC承接人数占比。
每一行是一个项目，项目由mssc组织或者代表处的人承接。MSSC承接人数占比是MSSC承接人数占总结承接人数的占比。
代表处是下对应一个国家的名字，代表这个国家下的项目"""
    question = """你需要分析mssc按承接占比波动的原因，它有3种情况：
    1.mssc在25年按承接占比高于24，他可能是代表处人数减少，项目工作量增加，
    2.mssc在25年按承接占比低于24，他可能是代表处人数增加，项目工作量减少，
    3.其他情况，暂时不分析
    比如：
    某个代表处2025年mssc承接人数减少，代表处人数减少，mssc承接占比减少，说明整体工作量下降。
    """
    ans = ChainOfThoughtAgent().generate_steps(df, df_des, question)
    print(ans)
    rlts = {}
    for i in ans:
        rlt = CodeExecutorAgent().run(i, df, df_des, max_retries=3)
        rlts[i] = rlt["result"]
    print(rlts)
    summary_agent = SummaryAgent()
    ans = summary_agent.summarize(df_des, df, question, rlts)
    print("ans",ans)