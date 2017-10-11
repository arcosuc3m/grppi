# chunk_sum

This example generates a sequence where each element is the sum of a windows of
values in the sequence of natural numbers

## Arguments

The program takes the following arguments:

* **size**: Number of elements initially generated.
* **window_size**: Size of the aggregation window.
* **offset**: Distance between the start of two consecutive windows.
* **mode**: Execution mode

## Behaviour

The program initially generates a stream of **size** consecutive numbers
starting from `0`.

Then, every **offset** elements an aggregation window of **window_size* is
started.

The aggregation method is the addition of all elements in the _window_.

## Examples

---
**Example 1**: Addition of consecutive non-overlapping pairs

0+1, 2+3, 4+5, ...
~~~
$ chunk_sum 10 2 2 seq
1
5
9
13
17
~~~
---

---
**Example 2**: Addition of consecutive overlapping pairs

0+1, 1+2, 2+3, 3+4, ...
~~~
$ chunk_sum 10 2 1 seq
1
3
5
7
9
11
13
15
17
~~~
---

---
**Example 3**: Addition of windows of 5 elements with offset of 2

0+1+2+3+4, 2+3+4+5+6, 4+5+6+7+8, 6+7+8+9
~~~
$ chunk_sum 10 5 2 seq
10
20
30
30
~~~
---
